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
from .filter_utils import _get_prior_spatial_points, uncertain_inputs_fitc_sparsity_compute_psi_statistics, _setup_state_and_args_dict
from .kalman_filter import _kalman_predict


# Import types
from ...transforms.sdes import SDE, LTI_SDE, LinearizedFilter_SDE
from ...transforms.pdes import PDE
from ...transforms.uncertain_inputs import UncertainPredictionInput


import objax
import chex

@dispatch(LinearizedFilter_SDE)
def get_model_H(prior, state, filter_res, args, m, P, m_predicted, P_predicted, X_s, t, full_state):
    # force full state
    H_k = np.eye(x.shape[0]) 

    return H_k

@dispatch(LTI_SDE)
def get_model_H(prior, state, filter_res, args, m, P, m_predicted, P_predicted, X_s, t, full_state):
    if full_state:
        # force full state
        H_k = np.eye(x.shape[0])
    else:
        H_k = prior.H(None, X_s, t)

    return H_k

@dispatch('UncertainPredictionInput')
def get_model_H(prior, state, filter_res, args, m, P, m_predicted, P_predicted, X_s, t, full_state):
    base_prior = prior.parent
    rts_fn = evoke('get_model_H', base_prior)
    H_k =  rts_fn(base_prior, state, filter_res, args, m, P, m_predicted, P_predicted, X_s, t, full_state)
    # TODO: this needs to predict to f -- OR need to implement that properly
    
    pred_weights, pred_covar = uncertain_inputs_fitc_sparsity_compute_psi_statistics(prior, args, m_predicted, P_predicted, H_k, X_s)

    H_k =  pred_weights @ H_k

    return H_k, pred_covar

@dispatch(PDE)
def get_model_H(prior, state, filter_res, args, m, P, m_predicted, P_predicted, X_s, t, full_state):
    H_parent = evoke('get_model_H', prior.parent)(prior, state, filter_res, args, m, P, m_predicted, P_predicted, X_s, t, full_state)

    # why not use m, P here?
    if full_state:
        H1 = prior.H_full_state(m_predicted, X_s, t)
    else:
        #H1 = prior.H_jac(m_predicted, X_s, t) # computed Jacobian at m_predicted 
        #H1 = prior.H_jac(m_predicted, X_s, t) # computed Jacobian at m_predicted 
        H1 = H1 = prior.H(H_parent @ m_predicted, X_s, t) 


    # TODO: why is H1 from 11 x 21 not 11 x 
    print('H1: ', H1.shape, 'H_parent: ', H_parent.shape)
    return H_parent

def get_H(prior, state, filter_res, args, m, P, m_predicted, P_predicted, X_s, t, full_state):
    rts_fn = evoke('get_model_H', prior)
    return rts_fn(prior, state, filter_res, args, m, P, m_predicted, P_predicted, X_s, t, full_state)



@dispatch(LTI_SDE)
def apply_H(prior, state, state_args, m, P, X_s, t, full_state):
    if full_state:
        # force full state
        H_k = np.eye(m.shape[0])
    else:
        H_k = prior.H(None, X_s, t)

    return H_k @ m, H_k @ P @ H_k.T, H_k

@dispatch(LinearizedFilter_SDE)
def apply_H(prior, state, state_args, m, P, X_s, t, full_state):
    breakpoint()

@dispatch('UncertainPredictionInput')
def apply_H(prior, state, state_args, m, P, X_s, t, full_state):

    m_obs, P_obs, H_k = evoke('apply_H', prior.parent)(
        prior.parent, state, state_args, m, P, X_s, t, full_state
    )

    pred_weights, pred_covar = uncertain_inputs_fitc_sparsity_compute_psi_statistics(prior, state_args, m, P, H_k, X_s)

    return pred_weights @ m_obs, pred_weights @ P_obs @ pred_weights.T + pred_covar, pred_weights

@dispatch(PDE)
def apply_H(prior, state, state_args, m, P, X_s, t, full_state):
    m_obs, P_obs, H_parent = evoke('apply_H', prior.parent)(
        prior.parent, state, state_args, m, P, X_s, t, full_state
    )

    if full_state:
        H1 = prior.H_full_state(m, X_s, t)
    else:
        #H1 = prior.H_jac(m_predicted, X_s, t) # computed Jacobian at m_predicted 
        #H1 = prior.H_jac(m_predicted, X_s, t) # computed Jacobian at m_predicted 
        #H1 = prior.H(m, X_s, t) 
        pass

    return m_obs, P_obs, H_parent


def apply_H(prior, state, state_args, m, P, X_s, t, full_state):
    return evoke('apply_H', prior)(
        prior, state, state_args, m, P, X_s, t, full_state
    )

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
def rts_step_wrapper(prior, state, filter_res, args, X_s, Xs_prior, full_state):
    sde_prior = prior

    m_inf = state['m_inf']
    P_inf = state['P_inf']


    m_k = args['m']
    P_k = args['P']


    dt_k = args['dt']

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
        state['m'],
        state['P'],
        m_predicted,
        P_predicted,
        A_k, 
        Q_k

    )

    H_k = get_H(prior, state, filter_res, args, args['m'], args['P'], m_predicted, P_predicted, Xs_prior, args['t'], full_state)

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
def rts_step_wrapper(prior, state, filter_res, scan_args, X_s, Xs_prior, full_state):
    base_prior = prior.parent
    rts_fn = evoke('rts_step_wrapper', base_prior)
    m_res, p_res =  rts_fn(base_prior, state, filter_res, scan_args, X_s, Xs_prior, full_state)

    H_k, pred_var = get_H(prior, state, filter_res, scan_args, m_res['m'], m_res['P'], m_res['m'], m_res['P'], Xs_prior, scan_args['t'], full_state)

    p_res = {
        'm': H_k @ m_res['m'], 'P': H_k @ m_res['P'] @ H_k.T + pred_var
    }

    return m_res, p_res

@dispatch(PDE)
def rts_step_wrapper(prior, state, filter_res, args, X_s, Xs_prior, full_state):
    """
    Prior is either PDE[SDE[]] or PDE[UI[SDE]]
    """

    sde_prior = prior.parent

    if _ensure_str(sde_prior) == 'UncertainPredictionInput':
        sde_prior = sde_prior.parent

    m_inf = state['m_inf']
    P_inf = state['P_inf']

    dt_k = args['dt']

    if False:
        A_k = sde_prior.expm(Xs_prior, dt_k)
        Q_k = sde_prior.Q(dt_k, A_k, prior.P_inf(None, Xs_prior, None), Xs_prior)

        m_predicted = A_k @ args['m']
        P_predicted = A_k @ args['P'] @ A_k.T + Q_k

    m_k = args['m']
    P_k = args['P']

    m_predicted, P_predicted, A_k, Q_k = _kalman_predict(sde_prior, m_k, P_k, Xs_prior, dt_k, return_A_Q = True)

    m, P = rts_smoother_step(
        args['m'],
        args['P'],
        state['m'],
        state['P'],
        m_predicted,
        P_predicted,
        A_k, 
        Q_k
    )


    # TODO: why m_predicted?
    #H_k = get_H(prior, state, filter_res, args, args['m'], args['P'], m_predicted, P_predicted, Xs_prior, args['t'], full_state)
    m_obs, P_obs, _ = apply_H(prior, state, args, m, P, Xs_prior, args['t'], full_state)
    #H_k = get_H(prior, state, filter_res, args, args['m'], args['P'], m_predicted, P_predicted, Xs_prior, args['t'], full_state)

    state_res =  {
        'm': m, 'P': P, 'm_inf': m_inf, 'P_inf': P_inf
    }

    obs_res = {
        'm': m_obs, 'P': P_obs
    }

    return state_res, obs_res

def step_wrapper(data, m, filter_res, Xs_prior, full_state):
    """ Wrapper to support scan with rts_step """

    rts_fn = evoke('rts_step_wrapper', m)

    def _fn(state, args):
        return rts_fn(m, state, filter_res, args, data.X_space, Xs_prior, full_state)

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

    state_dict =  {
        'm': m_init,
        'P': P_init ,
        'm_inf': m_inf,
        'P_inf': P_inf
    }

    args_dict = {
        'm': filter_res['m'],
        'P': filter_res['P'],
        'dt': dt,
        't': X_t
    }


    # handle prior specific state and args
    state_dict, args_dict = _setup_state_and_args_dict(
        data, prior, m_inf, P_inf, Xs_prior, state_dict, args_dict, smoother=True
    )

    all_args_dict = args_dict # store for UI
    # remove last element as already correct
    args_dict = {key: np.flip(val, axis=0)[1:, ...] for key, val in args_dict.items()}

    state, ys = scan(
        jax.remat(step_wrap),
        state_dict,
        args_dict
    )

    m = ys['m']
    P = ys['P']

    # TODO: figure out what state and ys should be here
    if isinstance(prior, UncertainPredictionInput) or isinstance(prior.parent, UncertainPredictionInput) :
        init_args_dict = {key: val[0] for key, val in all_args_dict.items()}
        m_obs_init, P_obs_init, _ = apply_H(prior, None, init_args_dict, m_init, P_init, Xs_prior, X_t[0], full_state)
    else:
        m_obs_init, P_obs_init, _ = apply_H(prior, state, ys, m_init, P_init, Xs_prior, X_t[0], full_state)

    if settings.balance_state_space:
        A_last = prior.expm(Xs_prior, dt[-2]) #get the last step
        T_last, T_last_inf = get_ss_balance_transformation(A_last)  
        H_k = _transform_ss_H_param(T_last, T_last_inf, H_k)

    m = np.vstack([(m_obs_init)[None, ...], m])
    P = np.vstack([(P_obs_init)[None, ...], P])

    m = np.flip(m, axis=0)
    P = np.flip(P, axis=0)

    return m, P

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






