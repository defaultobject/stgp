
import jax
from jax import jacfwd, jit
import jax.numpy as np
from jax.lax import scan, associative_scan

from ... import settings 
from ..matrix_ops import cholesky, cholesky_solve, add_jitter, batched_balance_matrix, lti_disc, pad_by_repeat_last_elem
from ..gaussian import log_gaussian, log_gaussian_with_mask
from ...utils.nan_utils import get_same_shape_mask
from ...dispatch import dispatch, evoke

# Import types
from ...transforms.sdes import SDE, LTI_SDE

from .rts_smoother import get_H

import objax
import chex
import math

@jit
def _last_smoothing_element(F, Q, m , P):
    return np.zeros_like(P), m, P

@jit
def _generic_smoothing_element(F, Q, m , P):
    Pp = F @ P @ F.T + Q
    Pp_chol = cholesky(add_jitter(Pp, settings.jitter))
    E = cholesky_solve(Pp_chol, F @ P).T
    g = m - E @ F @ m
    L = P - E @ Pp @ E.T

    # FORCE PSD
    # required to fix very small errors that propogate
    L = 0.5 * (L + L.T)
    return E, g, L

@jit
def smoothing_operator(x1, x2):
    # opposite way around as we are in reverse
    # keep indexs this was so it matches the paper
    E_i, g_i, L_i = x2
    E_j, g_j, L_j = x1

    E = E_i @ E_j
    g = E_i @ g_j + g_i
    L = E_i @ L_j @ E_i.T +  L_i


    # FORCE PSD
    # required to fix very small errors that propogate
    L = 0.5 * (L + L.T)

    return E, g, L

def smoother_block(carry, state, X_s, prior):
    """
    The inputs has already been flipped. So references to `first_members' should be read as x_n (the final state)
    """
    x_first = carry['x_first']
    m_inf = carry['m_inf']
    P_inf = carry['P_inf']

    dt = state['dt']
    first_member_mask = state['first_member_mask']

    m_arr = state['m_arr']
    P_arr = state['P_arr']
    H_arr = state['H_arr']

    if settings.low_memory_mode:
         low_memory_wrapper = jax.remat
    else:
        low_memory_wrapper = lambda x: x

    A_arr = jax.vmap(low_memory_wrapper(lambda dt_k: prior.expm(X_s, dt_k)))(dt)
    Q_arr = jax.vmap(low_memory_wrapper(lambda dt_k, A_k: prior.Q(dt_k, A_k, P_inf, X_s)))(dt, A_arr)

    x_all = jax.vmap(
        low_memory_wrapper(_generic_smoothing_element)
    )(A_arr, Q_arr, m_arr, P_arr)


    # manualy set new x_last
    x_first_op = smoothing_operator(x_first, [x_all[i][0] for i in range(3)])

    # if first memebr use x_first, else use the updated one
    x_first = [x_first_op[i]*(1-first_member_mask[0]) + first_member_mask[0]*x_first[i] for i in range(3)]

    x_all = [
        x_all[i].at[0].set(x_first[i])
        for i in range(3)
    ]

    # TODO: double check that smoothing_operator is expecting the reversed list
    res = associative_scan(
        jax.vmap(low_memory_wrapper(smoothing_operator)), 
        x_all,
        reverse=False # do not need to reverse as already have reversed everything
    )

    m = res[1]
    P = res[2]

    #m = np.vstack([x_last[1][None, ...], m])
    #P = np.vstack([x_last[2][None, ...], P])

    #H_k = get_H(prior, None, None, X_s, X_t[0], full_state)

    # Extract orderded state
    m = jax.vmap(low_memory_wrapper(lambda H_k, m_k: H_k @ m_k))(H_arr, m)
    P = jax.vmap(low_memory_wrapper(lambda H_k, P_k: H_k @ P_k @ H_k.T))(H_arr, P)

    # update x_last
    next_x_first = [res[i][-1] for i in range(3)]

    carry = {
        'x_first': next_x_first, 
        'm_inf': m_inf, 
        'P_inf': P_inf
    }

    state['m'] = m
    state['P'] = P

    return carry, state



@dispatch('parallel')
def smoother(data, prior, filter_res, dt, X_t, X_s, full_state, is_prediction):
    m_last = filter_res['m'][-1]
    P_last = filter_res['P'][-1]

    m_arr = filter_res['m']
    P_arr = filter_res['P']

    P_inf = prior.P_inf(None, X_s, None)
    m_inf = prior.m_inf(None, X_s, None)
    H = prior.H(None, X_s, None)

    H_arr = np.tile(H[None, ...], [data.Nt, 1, 1])
    x_last = _last_smoothing_element(None, None, m_arr[-1], P_arr[-1])

    smoother_block_wrapper = lambda carry, state: smoother_block(carry, state, X_s, prior)

    dt_flip = np.flip(dt)
    H_arr_flip = np.flip(H_arr, axis=0)
    m_arr_flip = np.flip(m_arr, axis=0)
    P_arr_flip = np.flip(P_arr, axis=0)

    first_member_mask = np.hstack([np.array([1]), np.zeros(data.Nt-1)])


    if settings.low_memory_mode:
        # need to pad

        if settings.parallel_filter_block_size is None:
            block_size = 999
        else:
            block_size = settings.parallel_filter_block_size

        Nt = data.Nt
        num_blocks = math.ceil(Nt/block_size)
        pad_size = num_blocks*block_size-Nt

        # convert matrices to equal block sizes
        H_arr_flip = pad_by_repeat_last_elem(H_arr_flip, pad_size).reshape((num_blocks, block_size) + H_arr_flip.shape[1:])

        dt_flip = pad_by_repeat_last_elem(dt_flip[:, None], pad_size).reshape((num_blocks, block_size))

        m_arr_flip = pad_by_repeat_last_elem(m_arr_flip, pad_size).reshape((num_blocks, block_size) + m_arr_flip.shape[1:])
        P_arr_flip = pad_by_repeat_last_elem(P_arr_flip, pad_size).reshape((num_blocks, block_size) + P_arr_flip.shape[1:])

        first_member_mask = pad_by_repeat_last_elem(first_member_mask[:, None], pad_size).reshape((num_blocks, block_size))

        carry, state = jax.lax.scan(
            jax.remat(smoother_block_wrapper),
            {
                'x_first': list(x_last), 
                'm_inf': m_inf, 
                'P_inf': P_inf
            },
            {
                'dt': dt_flip,
                'H_arr': H_arr_flip,
                'first_member_mask': first_member_mask,
                'm_arr': m_arr_flip,
                'P_arr': P_arr_flip
            }
        )

        m = np.flip(np.vstack(state['m'])[:data.Nt, ...], axis=0)
        P = np.flip(np.vstack(state['P'])[:data.Nt, ...], axis=0)

    else:

        # TODO: reverse
        carry, state = smoother_block_wrapper(
            {
                'x_first': x_last, 
                'm_inf': m_inf, 
                'P_inf': P_inf
            },
            {
                'dt': dt_flip,
                'H_arr': H_arr_flip,
                'first_member_mask': first_member_mask,
                'm_arr': m_arr_flip,
                'P_arr': P_arr_flip
            }
        )
        m = np.flip(state['m'], axis=0)
        P = np.flip(state['P'], axis=0)

    return m, P

