""" 
Closely following 
    https://github.com/EEA-sensors/parallel-gps/blob/main/pssgp/kalman/parallel.py
    https://github.com/AaltoML/BayesNewton/blob/61cb0ebb23afb12de0008882bc3d16b864b7149e/bayesnewton/ops.py#L521
    https://github.com/tensorflow/probability/blob/399cfcb4edda192c9f0f070ac04035dcb0e5b3a5/tensorflow_probability/python/experimental/parallel_filter/parallel_kalman_filter_lib.py#L701
"""
import jax
from jax import jacfwd, jit
import jax.numpy as np
from jax.lax import scan, associative_scan
import math

from ... import settings 
from ..matrix_ops import cholesky, cholesky_solve, add_jitter, mat_inv, force_symmetric, solve_with_additive_inverse, get_tensor_memory_in_gb, pad_by_repeat_last_elem, balance_matrix, batched_balance_matrix, lti_disc
from ..gaussian import log_gaussian, log_gaussian_with_mask, log_gaussian_with_additive_precision_noise_with_mask
from ...utils.nan_utils import get_same_shape_mask
from ...dispatch import dispatch, evoke

# Import types
from ...transforms.sdes import SDE, LTI_SDE

from ..linalg import solve, solve_from_cholesky

import jax.scipy as jsp

import objax
import chex
from functools import partial

@jit
def fix_psd(A):
    return force_symmetric(A)

@jit
def _first_filtering_element_lik_precision(m, P, F, Q, H, R_inv, y):
    m_ = F @ m
    P_ = F @ P @ F.T + Q

    # S = H P_ H.T + R
    #   = T1 + R
    T1 =  H @ P_ @ H.T
    T = T1 @ R_inv + np.eye(T1.shape[0])

    R_inv_chol = cholesky(add_jitter(R_inv, settings.jitter))

    # K = P_ H.T S_inv
    #   =  (S_inv @ H @ P_).T
    #   =  ([H P_ H.T + R]^{-1} @ H @ P_).T
    K = solve_with_additive_inverse(T1, R_inv, H @ P_.T).T

    A = np.zeros_like(F)

    # b = m_ + K @ [y-Hm_]
    b = m_ + K @ (y - H @ m_)

    # C = P_ - K @ S @ K.T
    C = P_ - K @ T @ cholesky_solve(R_inv_chol,  K.T)

    # S_k - H Q H.T + R
    # eta = F.T H.T S_k^{-1} (y)
    # J = F.T H.T S_k^{-1} H F

    FH_S_inv = solve_with_additive_inverse(H @ Q @ H.T, R_inv, H @ F).T

    eta = FH_S_inv @ y
    J = FH_S_inv @ H @ F 

    C = force_symmetric(C)
    J = force_symmetric(J)

    return A, b, C, J, eta

@jit
def _first_filtering_element(m, P, F, Q, H, R, y):
    m_ = F @ m
    P_ = F @ P @ F.T + Q


    S1 = H @ P_ @ H.T + R
    #S1_chol = cholesky(S1)
    #K = cholesky_solve(S1_chol, H @ P_.T).T
    K = solve(S1, H @ P_.T).T

    A = np.zeros_like(F)
    b = m_ + K @ (y - H @ m_)
    C = P_ - K @ S1 @ K.T

    S = H @ Q @ H.T + R

    #S_chol = cholesky(S)
    #FH_S_inv = cholesky_solve(S_chol, H @ F).T
    FH_S_inv = solve(S, H @ F).T

    eta = FH_S_inv @ y
    J = FH_S_inv @ H @ F 

    C = force_symmetric(C)
    J = force_symmetric(J)

    return A, b, C, J, eta

@jit
def _first_filtering_element_nan(m, P, F, Q, H, R, y):
     
    A = np.zeros_like(F)
    b = m
    C = P
    eta = np.zeros_like(m)
    J = np.zeros_like(F)

    return A, b, C, J, eta

# Does not depend on R
_first_filtering_element_nan_lik_precision = _first_filtering_element_nan


@jit
def _generic_filtering_element_lik_precision(F, Q, H, R_inv, y):
    I = np.eye(F.shape[0])

    # S_k = H Q H.T + R
    # S_k = T1 + R
    T1 =  H @ Q @ H.T
    T = T1 @ R_inv + np.eye(T1.shape[0])

    # K = Q H.T S^{-1}
    K = solve_with_additive_inverse(T1, R_inv, H @ Q.T).T

    A = (I - K @ H) @ F
    b = K @ y
    C = (I - K @ H) @ Q

    # eta = F.T H.T S_k^{-1} (y)
    # J = F.T H.T S_k^{-1} H F

    FH_S_inv = solve_with_additive_inverse(H @ Q @ H.T, R_inv, H @ F).T

    eta = FH_S_inv @ y
    J = FH_S_inv @ H @ F 

    return A, b, C, J, eta

@jit
def _generic_filtering_element(F, Q, H, R, y):
    I = np.eye(F.shape[0])

    S = H @ Q @ H.T + R

    use_cholesky = False

    if use_cholesky:
        S_chol = cholesky(S)
        K = cholesky_solve(S_chol, H @ Q.T).T
    else:
        K = solve(S, H @ Q.T).T

    A = (I - K @ H) @ F
    b = K @ y
    C = (I - K @ H) @ Q

    if use_cholesky:
        eta = F.T @ H.T @ cholesky_solve(S_chol, y)
        J = F.T @ H.T @  cholesky_solve(S_chol, H @ F) 
    else:
        eta = F.T @ H.T @ solve(S, y)
        J = F.T @ H.T @  solve(S, H @ F) 

    return A, b, C, J, eta

@jit
def _generic_filtering_element_nan(F, Q, H, R, y):
    A = F
    b = np.zeros([F.shape[1], 1])
    C = Q
    J = np.zeros_like(Q)
    eta = np.zeros_like(b)

    return A, b, C, J, eta

# does not depend on R
_generic_filtering_element_nan_lik_precision = _generic_filtering_element_nan


@jit
def filtering_operator(x1, x2):
    """ combine individual elements """
    A_i, b_i, C_i, J_i, eta_i = x1
    A_j, b_j, C_j, J_j, eta_j = x2

    N, D = A_i.shape
    I = np.eye(D)

    if settings.parallel_kf_force_linear_solve:
        # scarifice some stability to be able to use CG and cholesky solves
        C_i_inv = solve(C_i, np.eye(C_i.shape[0]))
        inner_tmp = C_i_inv + J_j
        Aj_tmp = (solve(inner_tmp, A_j.T).T) @ C_i_inv

        A = Aj_tmp @ A_i
        C = Aj_tmp @ C_i @ A_j.T + C_j
        b = Aj_tmp @ (b_i + C_i @ eta_j)+b_j


        A_i_tmp = solve(inner_tmp.T, C_i_inv @ A_i).T

    else:
        inner_tmp = I+C_i@J_j
        #Aj_tmp = np.linalg.solve(inner_tmp.T, A_j.T).T
        Aj_tmp = jsp.linalg.solve(inner_tmp.T, A_j.T, assume_a='gen').T

        A = Aj_tmp @ A_i
        C = Aj_tmp @ C_i @ A_j.T + C_j
        b = Aj_tmp @ (b_i + C_i @ eta_j)+b_j

        inner_tmp = I+J_j@C_i
        #A_i_tmp = np.linalg.solve(inner_tmp.T, A_i).T
        A_i_tmp = jsp.linalg.solve(inner_tmp.T, A_i, assume_a='gen').T

    eta = A_i_tmp @ (eta_j - J_j @ b_i) + eta_i
    J = A_i_tmp @ J_j @ A_i + J_i

    # FORCE PSD
    # required to fix very small errors that propogate
    C = fix_psd(C) 
    J = fix_psd(J) 
    return A, b, C, J, eta

#@partial(jit, static_argnums=(3))
def filter_block(carry, state, X_s, prior):
    if settings.low_memory_mode:
         low_memory_wrapper = jax.remat
    else:
        low_memory_wrapper = lambda x: x

    x_0 = carry['x_0']
    m_inf = carry['m_inf']
    P_inf = carry['P_inf']
    dt = state['dt']
    first_member_mask = state['first_member_mask']

    A_arr = jax.vmap(low_memory_wrapper(lambda dt_k: prior.expm(X_s, dt_k)))(dt)
    Q_arr = jax.vmap(low_memory_wrapper(lambda dt_k, A_k: prior.Q(dt_k, A_k, P_inf, X_s)))(dt, A_arr)

    state_size = A_arr.shape[2]

    H_arr = state['H_arr']
    lik_mat_arr = state['lik_mat_arr']

    Y = state['Y']


    mask = get_same_shape_mask(Y)
    # collapse mask
    mask = np.any(np.any(mask, axis=1), axis=1).astype(int)

    Y_raw = Y
    Y = np.nan_to_num(Y)

    x_all = jax.vmap(
        low_memory_wrapper(_generic_filtering_element)
    )(A_arr, Q_arr, H_arr, lik_mat_arr, Y)

    x_all_nan = jax.vmap(
        low_memory_wrapper(_generic_filtering_element_nan)
    )(A_arr, Q_arr, H_arr, lik_mat_arr, Y)

    # combine nan and observered
    def get_mask(mask, r):
        return np.reshape(mask, [-1] + [1]*(len(r.shape)-1))

    x_all = [
        x_all[i] * get_mask(mask, x_all[i]) + x_all_nan[i] * (1-get_mask(mask, x_all[i]))
        for i in range(5)
    ]

    # manualy set new x_0
    x_0_op = filtering_operator(x_0, [x_all[i][0] for i in range(5)])
    x_0 = [x_0_op[i]*(1-first_member_mask[0]) + first_member_mask[0]*x_0[i] for i in range(5)]

    x_all = [
        x_all[i].at[0].set(x_0[i])
        for i in range(5)
    ]

    res = associative_scan(
        jax.vmap(low_memory_wrapper(filtering_operator)),
        x_all
    )
    
    # res[0] will be zero so only need to consider res[1] and res[2]

    filtered_means = np.vstack([m_inf[None, ...]*first_member_mask[0] + res[1][0]*(1-first_member_mask[0]), res[1][:-1]])
    filtered_cov = np.vstack([P_inf[None, ...]*first_member_mask[0] + res[2][0]*(1-first_member_mask[0]), res[2][:-1]])

    pred_means = jax.vmap(
        low_memory_wrapper(lambda H_k, m_k, F_k:  F_k @ m_k)
    ) (
        H_arr, filtered_means, A_arr
    ) 

    pred_cov = jax.vmap(
        low_memory_wrapper(lambda H_k, P_k, F_k, Q_k:  F_k @ P_k @ F_k.T  +  Q_k )
    )(H_arr, filtered_cov, A_arr, Q_arr)


    obs_means = jax.vmap(
        low_memory_wrapper(lambda H_k, m_k, F_k: H_k @ F_k @ m_k)
    ) (
        H_arr, filtered_means, A_arr
    ) 
    obs_pred_cov = jax.vmap(
        low_memory_wrapper(lambda H_k, P_k, F_k, Q_k: H_k @ F_k @ P_k @ F_k.T @ H_k.T  + H_k @ Q_k @ H_k.T)
    )(H_arr, filtered_cov, A_arr, Q_arr)

    # use Y_raw as masking is handled inside the vmap
    log_Z_k = jax.vmap(
        low_memory_wrapper(lambda Y_k, mu_k, S_k, R_k: np.sum(
            log_gaussian_with_mask(np.nan_to_num(Y_k), mu_k, S_k+R_k, get_same_shape_mask(Y_k)[:, 0])
        )
    )) (Y_raw, obs_means, obs_pred_cov, lik_mat_arr)


    # x_0 for the next batch is the last m_, P_
    next_x_0 = [res[i][-1] for i in range(5)]

    carry = {
        'x_0': next_x_0, 
        'm_inf': m_inf, 
        'P_inf': P_inf
    }

    state['log_Z'] = log_Z_k
    state['filtered_mean'] = filtered_means
    state['filtered_cov'] = filtered_cov
    state['m'] = res[1]
    state['P'] = res[2]
    state['x_0'] = next_x_0


    return carry, state

@dispatch('parallel')
def _filter(data, prior, lik_mat, Y, X_t, X_s, dt, lik_cov_flag, train_test_mask, train_index):

    # compute steady states
    P_inf = prior.P_inf(None, X_s, None)
    m_inf = prior.m_inf(None, X_s, None)
    H = prior.H(None, X_s, None)

    # precompute all filtering parameters
    # TODO: this is O(N_t)!! need to distribute
    #dt = np.ones(Y.shape[0])*dt[1]
    lik_mat_arr = lik_mat
    A_arr = jax.vmap(lambda dt_k: prior.expm(X_s, dt_k))(dt)
    Q_arr = jax.vmap(lambda dt_k, A_k: prior.Q(dt_k, A_k, P_inf, X_s))(dt, A_arr)
    H_arr = np.tile(H[None, ...], [lik_mat_arr.shape[0], 1, 1])

    if settings.balance_state_space:
        # balance
        d = batched_balance_matrix(A_arr, settings.balance_state_space_iters)
        D = jax.vmap(np.diag)(d)
        d_inv = 1/d
        D_inv = jax.vmap(np.diag)(d_inv)

        A_arr = D_inv @ A_arr @ D
        Q_arr = D_inv @ Q_arr @ D_inv
        #P_inf = D_inv[0] @ P_inf @ D[0]
        P_inf = D_inv[0] @ P_inf @ D_inv[0]
        m_inf = D[0] @ m_inf 
        H_arr =  H_arr @ D

        #Q_arr = jax.vmap(lambda A_k: P_inf - A_k @ P_inf @ A_k.T)(A_arr)

    mask = get_same_shape_mask(Y)
    # collapse mask
    mask = np.any(np.any(mask, axis=1), axis=1).astype(int)

    Y = np.nan_to_num(Y)

    if lik_cov_flag:
        # lik_mat is a covariance
        x_0 = _first_filtering_element(m_inf, P_inf, A_arr[0], Q_arr[0], H, lik_mat_arr[0], Y[0])
        x_0_nan = _first_filtering_element_nan(m_inf, P_inf, A_arr[0], Q_arr[0], H, lik_mat_arr[0], Y[0])
    else:
        # lik_mat is a precision
        x_0 = _first_filtering_element_lik_precision(m_inf, P_inf, A_arr[0], Q_arr[0], H, lik_mat_arr[0], Y[0])
        x_0_nan = _first_filtering_element_nan_lik_precision(m_inf, P_inf, A_arr[0], Q_arr[0], H, lik_mat_arr[0], Y[0])

    # combine nan and observered
    x_0 = [
        x_0[i] * mask[0] + x_0_nan[i] * (1-mask[0])
        for i in range(5)
    ]

    if lik_cov_flag:
        x_all = jax.vmap(
            _generic_filtering_element
        )(A_arr, Q_arr, H_arr, lik_mat_arr, Y)

        x_all_nan = jax.vmap(
            _generic_filtering_element_nan
        )(A_arr, Q_arr, H_arr, lik_mat_arr, Y)
    else:
        x_all = jax.vmap(
            _generic_filtering_element_lik_precision
        )(A_arr, Q_arr, H_arr, lik_mat_arr, Y)

        x_all_nan = jax.vmap(
            _generic_filtering_element_nan_lik_precision
        )(A_arr, Q_arr, H_arr, lik_mat_arr, Y)

    # combine nan and observered
    def get_mask(mask, r):
        return np.reshape(mask, [-1] + [1]*(len(r.shape)-1))

    x_all = [
        x_all[i] * get_mask(mask, x_all[i]) + x_all_nan[i] * (1-get_mask(mask, x_all[i]))
        for i in range(5)
    ]

    x_all = [
        x_all[i].at[0].set(x_0[i])
        for i in range(5)
    ]

    res = associative_scan(
        jax.vmap(filtering_operator), 
        x_all
    )

    filtered_means = np.vstack([m_inf[None, ...], res[1][:-1]])
    filtered_cov = np.vstack([P_inf[None, ...], res[2][:-1]])

    obs_means = jax.vmap(
        lambda H_k, m_k, F_k: H_k @ F_k @ m_k
    ) (
        H_arr, filtered_means, A_arr
    ) 
    obs_pred_cov = jax.vmap(
        lambda H_k, P_k, F_k, Q_k: H_k @ F_k @ P_k @ F_k.T @ H_k.T  + H_k @ Q_k @ H_k.T
    )(H_arr, filtered_cov, A_arr, Q_arr)


    if lik_cov_flag:
        log_Z_k = jax.vmap(
            lambda Y_k, mu_k, S_k, R_k: np.sum(
                log_gaussian_with_mask(np.nan_to_num(Y_k), mu_k, S_k+R_k, get_same_shape_mask(Y_k)[:, 0])
            )
        ) (Y, obs_means, obs_pred_cov, lik_mat_arr)
    else:
        log_Z_k = jax.vmap(
            lambda Y_k, mu_k, S_k, R_inv_k: np.sum(
                log_gaussian_with_additive_precision_noise_with_mask(
                    np.nan_to_num(Y_k), mu_k, S_k, R_inv_k, get_same_shape_mask(Y_k)[:, 0]
                )
            )
        ) (Y, obs_means, obs_pred_cov, lik_mat_arr)

        tmp_log_Z_k = jax.vmap(
            lambda Y_k, mu_k, S_k, R_k: np.sum(
                log_gaussian_with_mask(np.nan_to_num(Y_k), mu_k, S_k+R_k, get_same_shape_mask(Y_k)[:, 0])
            )
        ) (Y, obs_means, obs_pred_cov, jax.vmap(mat_inv)(lik_mat_arr))
        
    log_Z = np.sum(log_Z_k)

    return log_Z, {'m': res[1], 'P': res[2]}

@dispatch('parallel')
def filter(data, prior, lik_mat, Y, X_t, X_s, dt, lik_cov_flag, train_test_mask, train_index):

    if lik_cov_flag is False:
        raise NotImplementedError()

    if settings.low_memory_mode:
         low_memory_wrapper = jax.remat
    else:
        low_memory_wrapper = lambda x: x

    # compute steady states
    P_inf = prior.P_inf(None, X_s, None)
    m_inf = prior.m_inf(None, X_s, None)
    H = prior.H(None, X_s, None)

    # precompute all filtering parameters
    # TODO: this is O(N_t)!! need to distribute
    #dt = np.ones(Y.shape[0])*dt[1]
    lik_mat_arr = lik_mat

    # compute initial parameters
    A_0 = prior.expm(X_s, dt[0])
    Q_0 = prior.Q(dt[0], A_0, P_inf, X_s)

    H_arr = np.tile(H[None, ...], [lik_mat_arr.shape[0], 1, 1])

    # constrct nan masks
    mask = get_same_shape_mask(Y)
    # collapse mask
    mask = np.any(np.any(mask, axis=1), axis=1).astype(int)
    # lik_mat is a covariance
    x_0 = _first_filtering_element(m_inf, P_inf, A_0, Q_0, H, lik_mat_arr[0], np.nan_to_num(Y[0]))
    x_0_nan = _first_filtering_element_nan(m_inf, P_inf, A_0, Q_0, H, lik_mat_arr[0], np.nan_to_num(Y[0]))

    # combine nan and observered
    x_0 = [
        x_0[i] * mask[0] + x_0_nan[i] * (1-mask[0])
        for i in range(5)
    ]

    filter_block_wrapper = lambda carry, state: filter_block(carry, state, X_s, prior)

    # create mask for first element
    first_member_mask = np.hstack([np.array([1]), np.zeros(data.Nt-1)])

    if settings.low_memory_mode:
        Nt = data.Nt
        if settings.parallel_filter_block_size is None:
            block_size = 999
        else:
            block_size = settings.parallel_filter_block_size

        num_blocks = math.ceil(Nt/block_size)
        pad_size = num_blocks*block_size-Nt

        # convert matrices to equal block sizes
        H_arr = pad_by_repeat_last_elem(H_arr, pad_size).reshape((num_blocks, block_size) + H_arr.shape[1:])
        lik_mat_arr = pad_by_repeat_last_elem(lik_mat_arr, pad_size).reshape((num_blocks, block_size) + lik_mat_arr.shape[1:])
        Y = pad_by_repeat_last_elem(Y, pad_size, pad_with_nan=True).reshape((num_blocks, block_size) + Y.shape[1:])

        dt = pad_by_repeat_last_elem(dt[:, None], pad_size).reshape((num_blocks, block_size))

        first_member_mask = pad_by_repeat_last_elem(first_member_mask[:, None], pad_size).reshape((num_blocks, block_size))

        # TODO: scan over first dimension
        carry, state = jax.lax.scan(
            jax.remat(filter_block_wrapper),
            {
                'x_0': x_0, 
                'm_inf': m_inf, 
                'P_inf': P_inf
            },
            {
                'lik_mat_arr': lik_mat_arr,
                'Y': Y,
                'dt': dt,
                'H_arr': H_arr,
                'first_member_mask': first_member_mask
            }
        )

        log_Z = np.sum(np.vstack(state['log_Z'])[:data.Nt]) # only keep the unpadded log_Z
        filtered_means =  np.vstack(state['filtered_mean'])
        filtered_covs =  np.vstack(state['filtered_cov'])

        # remove padded elements
        filtered_means = filtered_means[:data.Nt, ...]
        filtered_covs = filtered_covs[:data.Nt, ...]
        res = [filtered_means, filtered_covs]
    else:
        carry, state = filter_block_wrapper(
            {
                'x_0': x_0, 
                'm_inf': m_inf, 
                'P_inf': P_inf
            },
            {
                'lik_mat_arr': lik_mat_arr,
                'Y': Y,
                'dt': dt,
                'H_arr': H_arr,
                'first_member_mask': first_member_mask
            }
        )
        log_Z = np.sum(state['log_Z'])
        #res = [state['filtered_mean'], state['filtered_cov']]
        res = [state['m'], state['P']]

    return log_Z, {'m': res[0], 'P': res[1]}

