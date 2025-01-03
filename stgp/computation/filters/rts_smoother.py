"""
Sequential Kalman Smoother

Models:
    FITCSpatialSparsity:
        Does not require any special cases here as the spatial predictions are handled in the SDE_GP class
        ie we do no need to predict to the training data here, as we will be predicting to the test data anyway
            downstream
"""
import jax
from jax import jacfwd, jit
import jax.numpy as np
from jax.lax import scan

from ... import settings 
from ..matrix_ops import cholesky, cholesky_solve, add_jitter, batched_balance_matrix
from ..gaussian import log_gaussian, log_gaussian_with_mask
from ...utils.nan_utils import get_same_shape_mask
from ...dispatch import dispatch, evoke, _ensure_str
from ...computation.model_ops import get_ss_balance_transformation, _transform_ss_A_param, _transform_ss_H_param, _transform_ss_init_params, _transform_ss_Q_param
from .filter_utils import _get_prior_spatial_points
from .kalman_filter import _kalman_predict


# Import types
from ...transforms.sdes import SDE, LTI_SDE, LinearizedFilter_SDE
from ...transforms.pdes import PDE


import objax
import chex

@dispatch(LinearizedFilter_SDE)
def get_model_H(prior, filter_res, x, m_predicted, X_s, t, full_state):
    # force full state
    H_k = np.eye(x.shape[0]) 

    return H_k

@dispatch(LTI_SDE)
def get_model_H(prior, filter_res, x, m_predicted, X_s, t, full_state):
    if full_state:
        # force full state
        H_k = np.eye(x.shape[0])
    else:
        H_k = prior.H(None, X_s, t)

    return H_k

@dispatch('UncertainPredictionInput')
def get_model_H(prior, filter_res, x, m_predicted, X_s, t, full_state):
    base_prior = prior.parent
    rts_fn = evoke('get_model_H', base_prior)
    H_k =  rts_fn(base_prior, filter_res, x, m_predicted, X_s, t, full_state)
    # TODO: this needs to predict to f -- OR need to implement that properly
    return H_k

@dispatch(PDE)
def get_model_H(prior, filter_res, x, m_predicted, X_s, t, full_state):
    H_parent = evoke('get_model_H', prior.parent)(prior, filter_res, x, m_predicted, X_s, t, full_state)


    if full_state:
        H1 = prior.H_full_state(m_predicted, X_s, t)
    else:
        #H1 = prior.H_jac(m_predicted, X_s, t) # computed Jacobian at m_predicted 
        #H1 = prior.H_jac(m_predicted, X_s, t) # computed Jacobian at m_predicted 
        H1 = H1 = prior.H(H_parent @ m_predicted, X_s, t) 


    # TODO: why is H1 from 11 x 21 not 11 x 
    print('H1: ', H1.shape, 'H_parent: ', H_parent.shape)
    return H_parent

def get_H(prior, filter_res, x, m_predicted, X_s, t, full_state):
    rts_fn = evoke('get_model_H', prior)
    return rts_fn(prior, filter_res, x, m_predicted, X_s, t, full_state)

@jit
def rts_smoother_step(m_filtered_k, P_filtered_k, m, P, m_predicted, P_predicted, A_k, Q_k):
    """
    Computes the RTS smoother step:

         
    """
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
def rts_step_wrapper(prior, carry, filter_res, x, X_s, Xs_prior, full_state):
    sde_prior = prior

    m_inf = carry['m_inf']
    P_inf = carry['P_inf']


    m_k = x['m']
    P_k = x['P']


    dt_k = x['dt']

    m_predicted, P_predicted, A_k, Q_k = _kalman_predict(prior, m_k, P_k, Xs_prior, dt_k, return_A_Q = True)

    if False:
        A_k = sde_prior.expm(Xs_prior, dt_k)
        
        # Compute Q_k in the untransformed space
        Q_k = sde_prior.Q(dt_k, A_k, prior.P_inf(None, Xs_prior, None), Xs_prior)
        if settings.balance_state_space:
            T_k, T_k_inf = get_ss_balance_transformation(A_k)
            _A_k = A_k
            _Q_k = Q_k
            A_k = _transform_ss_A_param(T_k, T_k_inf, A_k)
            Q_k = _transform_ss_Q_param(T_k, T_k_inf, Q_k)


        m_predicted = A_k @ m_k
        P_predicted = A_k @ P_k @ A_k.T + Q_k

    m, P = rts_smoother_step(
        m_k,
        P_k,
        carry['m'],
        carry['P'],
        m_predicted,
        P_predicted,
        A_k, 
        Q_k

    )

    H_k = get_H(prior, filter_res, x['m'], m_predicted, Xs_prior, x['t'], full_state)

    if settings.balance_state_space:
        H_k = _transform_ss_H_param(T_k, T_k_inf, H_k)

    m_res =  {
        'm': m, 'P': P, 'm_inf': m_inf, 'P_inf': P_inf
    }

    p_res = {
        'm': H_k @ m, 'P': H_k @ P @ H_k.T
    }


    return m_res, p_res


@dispatch('UncertainPredictionInput')
def rts_step_wrapper(prior, carry, filter_res, x, X_s, Xs_prior, full_state):
    base_prior = prior.parent
    rts_fn = evoke('rts_step_wrapper', base_prior)
    return rts_fn(base_prior, carry, x, X_s, Xs_prior, full_state)

@dispatch(PDE)
def rts_step_wrapper(prior, carry, filter_res, x, X_s, Xs_prior, full_state):
    """
    Prior is either PDE[SDE[]] or PDE[UI[SDE]]
    """

    sde_prior = prior.parent

    if _ensure_str(sde_prior) == 'UncertainPredictionInput':
        sde_prior = sde_prior.parent

    m_inf = carry['m_inf']
    P_inf = carry['P_inf']

    dt_k = x['dt']

    if False:
        A_k = sde_prior.expm(Xs_prior, dt_k)
        Q_k = sde_prior.Q(dt_k, A_k, prior.P_inf(None, Xs_prior, None), Xs_prior)

        m_predicted = A_k @ x['m']
        P_predicted = A_k @ x['P'] @ A_k.T + Q_k

    m_k = x['m']
    P_k = x['P']

    m_predicted, P_predicted, A_k, Q_k = _kalman_predict(sde_prior, m_k, P_k, Xs_prior, dt_k, return_A_Q = True)

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

    H_k = get_H(prior, filter_res, x['m'], m_predicted, Xs_prior, x['t'], full_state)

    m_res =  {
        'm': m, 'P': P, 'm_inf': m_inf, 'P_inf': P_inf
    }

    p_res = {
        'm': H_k @ m, 'P': H_k @ P @ H_k.T
    }

    return m_res, p_res

def step_wrapper(data, m, filter_res, Xs_prior, full_state):
    """ Wrapper to support scan with rts_step """

    rts_fn = evoke('rts_step_wrapper', m)

    def _fn(carry, x):
        return rts_fn(m, carry, filter_res, x, data.X_space, Xs_prior, full_state)

    return _fn

@dispatch('sequential')
def smoother(data, prior, filter_res, dt, X_t, Xs_prior, full_state):
    m_init = filter_res['m'][-1]
    P_init = filter_res['P'][-1]

    m_inf = prior.m_inf(None, Xs_prior, None)
    P_inf = prior.P_inf(None, Xs_prior, None)

    if settings.balance_state_space:
        A_0 = prior.expm(Xs_prior, 0.0) # steady state has no dt
        T, T_inf = get_ss_balance_transformation(A_0)  
        m_inf, P_inf = _transform_ss_init_params(T, T_inf, m_inf, P_inf)

    step_wrap = step_wrapper(data, prior, filter_res, Xs_prior, full_state)

    carry, ys = scan(
        jax.remat(step_wrap),
        {
            'm': m_init,
            'P': P_init ,
            'm_inf': m_inf,
            'P_inf': P_inf
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

    H_k = get_H(prior, filter_res, m_init, m_init, Xs_prior, X_t[0], full_state)

    if settings.balance_state_space:
        A_last = prior.expm(Xs_prior, dt[-2]) #get the last step
        T_last, T_last_inf = get_ss_balance_transformation(A_last)  
        H_k = _transform_ss_H_param(T_last, T_last_inf, H_k)

    m = np.vstack([(H_k @ m_init)[None, ...], m])
    P = np.vstack([(H_k @ P_init @ H_k.T)[None, ...], P])

    print('smoother: ', m.shape, P.shape)

    return np.flip(m, axis=0), np.flip(P, axis=0)

def smoother_loop(data: 'SequentialData', prior: 'Prior', filter_res: dict, full_state=False, filter_type=False):
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

    # spatial points that the prior is defined over
    Xs_prior = _get_prior_spatial_points(data, prior)

    if settings.verbose:
        print(f'running {filter_type} kalman smoother')

    # sequential, parallel, square_root_svm
    smoother_fn = evoke('smoother', filter_type)

    mu, var  =  smoother_fn(data, prior, filter_res, dt, X_t, Xs_prior, full_state)

    return mu, var






