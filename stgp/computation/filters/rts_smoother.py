import jax
from jax import jacfwd, jit
import jax.numpy as np
from jax.lax import scan

from ... import settings 
from ..matrix_ops import cholesky, cholesky_solve, add_jitter
from ..gaussian import log_gaussian, log_gaussian_with_mask
from ...utils.nan_utils import get_same_shape_mask
from ...dispatch import dispatch, evoke

# Import types
from ...transforms.sdes import SDE, LTI_SDE

import objax
import chex

@jit
def rts_smoother_step(m_filtered_k, P_filtered_k, m, P, m_predicted, P_predicted, A_k, Q_k):
    """
    Computes the RTS smoother step:

    TODO
         
    """
    #kalman gain
    P_predicted_chol = cholesky(
        add_jitter(P_predicted, settings.jitter)
    )
    G = cholesky_solve(
        P_predicted_chol, A_k @ P_filtered_k
    ).T

    m = m_filtered_k + G @ (m - m_predicted)
    P = P_filtered_k + G @ (P - P_predicted) @ G.T

    return m, P

@dispatch(LTI_SDE)
def rts_step(prior, carry, x, X_s, full_state=False):
    P_inf = prior.P_inf(None, X_s, None)
    H_k = prior.H(None, X_s, None)

    dt_k = x['dt']

    A_k = prior.expm(X_s, dt_k)
    Q_k = P_inf - A_k @  P_inf @ A_k.T

    m_predicted = A_k @ x['m']
    P_predicted = A_k @ x['P'] @ A_k.T + Q_k

    m, P = rts_smoother_step(
        x['m'],
        x['P'],
        carry['m'],
        carry['P'],
        m_predicted,
        P_predicted,
        A_k, 
        Q_k

    )
    m_res =  {
        'm': m, 'P': P 
    }

    if full_state:
        p_res = {
            'm': m, 'P':  P 
        }
    else:
        p_res = {
            'm': H_k @ m, 'P': H_k @ P @ H_k.T
        }

    return m_res, p_res

@dispatch(SDE)
def rts_step(model, carry, x, X_s, full_state=False):
    """ Extended Kalman Filter Predict Step """
    H_k = model.H(None, X_s, None)

    f_fn = lambda m: model.f_dt(
        m, X_s, x['t'], x['dt']
    )
    Sigma = model.Sigma_dt(
        carry['m'], X_s, x['t'], x['dt']
    )

    f = f_fn(x['m'])
    F = jax.jacfwd(f_fn)(x['m'])
    F = F[:, 0, :, 0]

    m, P = rts_smoother_step(
        x['m'],
        x['P'],
        carry['m'],
        carry['P'],
        f,
        F @ x['P'] @ F.T + Sigma,
        F, 
        Sigma

    )
    m_res =  {
        'm': m, 'P': P 
    }

    if full_state:
        p_res = {
            'm': m, 'P':  P 
        }
    else:
        p_res = {
            'm': H_k @ m, 'P': H_k @ P @ H_k.T
        }

    return m_res, p_res

def step_wrapper(data, m, full_state=False):
    rts_fn = evoke('rts_step', m)

    def _fn(carry, x):
        return rts_fn(m, carry, x, data.X_space, full_state=full_state)

    return _fn

def smoother_loop(data: 'SequentialData', model: 'Model', filter_res: dict, full_state=False):
    """
    Args:
        full_state: flag -- if False we only return part of the state corresponding to the latent GP, else returns the whole state
    """
    # Set up data
    X_t = data.X_time
    X_s =  data.X_space

    N_t = data.Nt
    N_s = data.Ns
    P = data.P

    out_dim = N_s * P

    dt = np.diff(X_t)
    # TODO: fix this
    dt = np.hstack([dt, np.zeros(1)])

    step_wrap = step_wrapper(data, model, full_state=full_state)
    H_k = model.H(None, X_s, None)

    m_init = filter_res['m'][-1]
    P_init = filter_res['P'][-1]

    carry, ys = scan(
        step_wrap,
        {
            'm': m_init,
            'P': P_init 
        },
        {
            'm': np.flip(filter_res['m'], axis=0)[1:, ...],
            'P': np.flip(filter_res['P'], axis=0)[1:, ...],
            'dt': np.flip(dt, axis=0)[1:, ...],
            't': np.flip(X_t, axis=0)[1:, ...]
        }
    )

    m = ys['m']
    P = ys['P']

    if full_state:
        m = np.vstack([(m_init)[None, ...], m])
        P = np.vstack([(P_init)[None, ...], P])

    else:
        m = np.vstack([(H_k @ m_init)[None, ...], m])
        P = np.vstack([(H_k @ P_init @ H_k.T)[None, ...], P])

    return np.flip(m, axis=0), np.flip(P, axis=0)


