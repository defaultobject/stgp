"""
Kalman filtering

With one latent function the full state has the following format:
    time-space-state

With multiple latent functions the full state is:
    time-latent-space-state

because this corresponds to simply stacking the latent GPs

when using derivate observations the state is organised as:
    time-latent-ds-space-df
    

"""
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
def kf_update_step(m_, P_, H_k, R_k, carry, x):
    """
    Computes the Kalman filter update equations with missing data support:
    
    In: 
        p(x_k | Y_{k-1}) = N(x_k | _m_k, _P_k)

    Computes:

        v_k = y_k - H_k - _m_k 
        S_k = H_k _P_k H^T_k + R_k
        K_k = _P_k H^T_k S^{-1}_k

        m_k = _m_k + K_k v_k
        P_k = _P_k - K_k S_k K^T_k

    TODO:
    Args:
        carry:
        x:
    """

    # in latent - space format
    Y_k = x['Y']

    mask_k = get_same_shape_mask(Y_k)

    Y_k = np.nan_to_num(Y_k)

    # Construct spatial mask
    m_vec = np.tile(mask_k, [1, Y_k.shape[0]])

    M = np.multiply(
        m_vec,
        np.eye(Y_k.shape[0])
    )

    # -- KALMAN UPDATE --
    # m_, P_ is in latent - space -state format
    # convert to latent-space format

    mu = M @ H_k @ m_
    var = M @ H_k @ P_ @ H_k.T @ M.T

    #inovation mean and variance
    # all in latent-space format
    v = Y_k - mu
    S = var + R_k

    #Kalman Gain
    S = add_jitter(S, settings.jitter)
    L = cholesky(S)
    K = (cholesky_solve(L, M @ H_k @ P_)).T

    # Kalman Update
    # convert to latent-space-state format before updating
    m_k = m_ + K @ v
    P_k = P_ - K @ S @ K.T

    #log marginal likelihood (assuming Gaussian likelihood)
    log_Z_k = np.sum(
        log_gaussian_with_mask(Y_k, mu, S, mask_k[:, 0])
    )

    return {
        'm': m_k, 'P': P_k 
    }, {
        'm': m_k, 'P': P_k, 'lml': log_Z_k
    }


@dispatch(LTI_SDE, 'sequential')
def kf_predict_step(prior, carry, x, X_s):
    """ Linear Kalman Filter Predict Step """

    P_inf = prior.P_inf(None, X_s, None)
    H_k = prior.H(None, X_s, None)

    R_k =  x['R']
    dt_k = x['dt']

    m_k = carry['m']
    P_k = carry['P']

    A_k = prior.expm(X_s, dt_k)
    Q_k = P_inf - A_k @  P_inf @ A_k.T

    m_ = A_k @ m_k
    P_ = A_k @ P_k @ A_k.T + Q_k

    return kf_update_step(m_, P_, H_k, R_k, carry, x)


@dispatch(SDE, 'sequential')
def kf_predict_step(model, carry, x, X_s):
    """ Extended Kalman Filter Predict Step """
    H_k = model.H(None, X_s, None)
    R_k =  x['R']

    m = carry['m']
    P = carry['P']

    f_fn = lambda m: model.f_dt(
        m, X_s, x['t'], x['dt']
    )

    f = f_fn(m)

    F = jax.jacfwd(f_fn)(m)
    F = F[:, 0, :, 0]

    Sigma = model.Sigma_dt(
        m, X_s, x['t'], x['dt']
    )

    m_ = f
    P_ = F @ P @ F.T + Sigma


    return kf_update_step(m_, P_, H_k, R_k, carry, x)


def filter_step_wrapper(data, m):
    kf_predict_fn = evoke('kf_predict_step', m, 'sequential')


    def _fn(carry, x):
        return kf_predict_fn(m, carry, x, data.X_space)

    return _fn

@dispatch('sequential')
def filter(data, prior, R, Y, X_t, X_s, dt):

    # steady state does not depend on time
    # in latent-space-state format
    m_inf = prior.m_inf(None, X_s, None)
    P_inf = prior.P_inf(None, X_s, None)

    step_wrap = filter_step_wrapper(data, prior)

    carry, ys = scan(
        step_wrap,
        {
            'm': m_inf,
            'P': P_inf
        },
        {
            'dt': dt,
            't': X_t,
            'Y': Y,
            'R': R
        }
    )

    lml = np.sum(ys['lml'])

    filter_res = {'m': ys['m'], 'P': ys['P']}

    return lml, filter_res

def filter_loop(data: 'SequentialData', prior: 'Prior', R, parallel=False):
    """
    Args:
        R: in time - latent - space format
    """ 

    x_t =  data.X_time
    X_s =  data.X_space

    N_t = data.Nt
    N_s = data.Ns
    P = data.P

    out_dim = N_s * P

    # Set up data
    X_t = data.X_time
    Y = data.Y_st
    dt = np.diff(X_t)

    # TODO: fix this
    dt = np.hstack([np.ones(1), dt])

    # Fix Y shapeo
    Nt = data.Nt
    Ns = data.Ns
    P = data.P

    # Y has shape Nt x P x Ns
    chex.assert_shape(Y, [Nt, P, Ns])
    # flatten but still in time - latent - space format
    Y = np.reshape(Y, [data.Nt, -1])

    # Ensure rank 2 at each time step
    Y = Y[..., None]

    
    if parallel:
        filter_fn = evoke('filter', 'parallel')
    else:
        filter_fn = evoke('filter', 'sequential')

    lml, filter_res =  filter_fn(data, prior, R, Y, X_t, X_s, dt)


    return lml, filter_res

