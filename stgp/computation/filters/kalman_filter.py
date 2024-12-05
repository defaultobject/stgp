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
from functools import partial

from ... import settings 
from ..matrix_ops import cholesky, cholesky_solve, add_jitter, mat_inv, solve_with_additive_inverse, force_symmetric, lti_disc
from ..gaussian import log_gaussian, log_gaussian_with_mask, log_gaussian_with_additive_precision_noise_with_mask, avg_mahal_with_mask, mahal_with_mask
from ...utils.nan_utils import get_same_shape_mask
from ...dispatch import dispatch, evoke, _ensure_str
from ...computation.model_ops import get_ss_balance_transformation, _transform_ss_params, _transform_ss_init_params, _transform_ss_Q_param
from ...computation.kernel_psi_statistics import get_fitc_sparsity_transformation, get_psi_statistics_linear_form_from_mu_var_from_gram
from ..predictors.base_predictors import gaussian_prediction_diagonal, gaussian_prediction_diagonal_statistics
from .filter_utils import _get_prior_spatial_points

import numpy as onp
#import tensorflow as tf
#import tensorflow_probability as tfp
#from tensorflow_probability.python.math.linalg import low_rank_cholesky, pivoted_cholesky
#from jax.experimental.jax2tf import call_tf

from ..linalg import solve, solve_from_cholesky

# Import types
from ...transforms.sdes import SDE, LTI_SDE, LinearizedFilter_SDE
from ...transforms.pdes import PDE
from ...transforms.uncertain_inputs import UncertainPredictionInput

import numpy as onp
import objax
import chex

def _state_block_dim(model, X_spatial):
    if X_spatial is None:
        Ns = 1
    else:
        Ns = X_spatial.shape[0]

    dt_dims = model.state_space_dim()
    ds_dims = model.spatial_output_dim

    if type(ds_dims) is list:
        if type(ds_dims[0]) is list:
            ds_dims = np.sum(ds_dims[0])
        else:
            ds_dims = ds_dims[0]

    if type(dt_dims) is list:
        if type(dt_dims[0]) is list:
            block_dim = sum(dt_dims[0])*ds_dims
        else:
            block_dim = dt_dims[0]*ds_dims
    else:
        block_dim = dt_dims*ds_dims

    block_dim = block_dim*Ns

    return block_dim

def _kalman_predict(prior, m_k, P_k, Xs_prior, dt_k):

    A_k_blocks = prior.expm_blocks(Xs_prior, dt_k)

    # use untransformed P_inf
    P_inf_blocks = prior.P_inf_blocks(None, Xs_prior, None) # should be block diagonal
    breakpoint()

    Q_k = prior.Q_blocks(dt_k, A_k_blocks, P_inf_blocks, X_spatial=Xs_prior) # will be block diagonal

    # convert to full matrices
    if False:
        A_k = to_block_diag(A_k_blocks)
        Q_k = to_block_diag(Q_k_blocks)

    if settings.balance_state_space:
        # balance_ss takes m_inf, P_inf but we don't need to pass those so pass dummy m_k, P_k
        T_k, T_k_inf = get_ss_balance_transformation(A_k)
        A_k, Q_k, H_k = _transform_ss_params(T_k, T_k_inf, A_k, Q_k, H_k)

    # predict steps
    m_ = A_k @ m_k
    P_ = A_k @ P_k @ A_k.T + Q_k

    return m_, P_

@jit
def kf_update_step_with_lik_precision(m_, P_, H_k, R_inv_k, carry, x):
    """
    Computes the Kalman filter update equations with missing data support:
    
    In: 
        p(x_k | Y_{k-1}) = N(x_k | _m_k, _P_k)


    Computes:

    Let R_k = R_inv_k^{-1} then 

        v_k = y_k - H_k - _m_k 
        S_k = H_k _P_k H^T_k + R_k
        K_k = _P_k H^T_k S^{-1}_k

        m_k = _m_k + K_k v_k
        P_k = _P_k - K_k S_k K^T_k

    Args:
        carry:
        x:
    """
    raise RuntimeError('NOT BEEN MAINTAINED')
    # in latent - space format
    Y_k = x['Y']

    mask_k = get_same_shape_mask(Y_k)

    Y_k = np.nan_to_num(Y_k)

    # Construct spatial mask
    m_vec = np.tile(mask_k, [1, Y_k.shape[0]])

    # only allow non-zero values through from non-nan values of Y
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

    #Kalman Gain
    #  K = P_ @ H.T @ S^{-1}
    #    = [S^{-1} @ (H @ P_.T)].T
    K = solve_with_additive_inverse(var, R_inv_k, M @ H_k @ P_).T

    R_inv_k_chol = cholesky(add_jitter(R_inv_k, settings.jitter))

    # Kalman Update
    # convert to latent-space-state format before updating
    m_k = m_ + K @ v
    # P_k = P - K @ S @ K.T
    #     = P - K @ [var + R_k] @ K.T
    #     = P - K @ [var R_k_inv + I] @ R_k @ K.T
    #     = P - K @ [var R_k_inv + I] @ [R_k_inv]^{-1} K.T

    A = var @ R_inv_k + np.eye(var.shape[0])
    A_chol = cholesky(A)

    P_k = P_ - K @ A @ cholesky_solve(R_inv_k_chol, K.T)

    P_k = force_symmetric(P_k)

    #log marginal likelihood (assuming Gaussian likelihood)
    log_Z_k = np.sum(
        log_gaussian_with_additive_precision_noise_with_mask(Y_k, mu, var, R_inv_k, mask_k[:, 0])
    )

    return {
        'm': m_k, 'P': P_k, 'm_inf': carry['m_inf'], 'P_inf': carry['P_inf']
    }, {
        'm': m_k, 'P': P_k, 'lml': log_Z_k
    }

def icholesky(H):
    """ https://github.com/google/jax/discussions/5068 """
    w, v = np.linalg.eigh(H)
    w = np.where(w < 0, 0.0001, w) # make this pd, psd is insufficient
    H_pd = v @ np.eye(3)*w @ v.T

    return jax.scipy.linalg.cholesky(H_pd), np.any(w < 0)

def _low_rank_cholesky(matrix):
    """ wrap low_rank_cholesky to make max_rank static """
    return low_rank_cholesky(matrix, onp.array(settings.cg_precondition_rank).astype(onp.int32))

def _pivoted_cholesky(matrix):
    """ wrap pivoted_cholesky to make max_rank static """
    return pivoted_cholesky(matrix, onp.array(settings.cg_precondition_rank).astype(onp.int32))

def get_Y_mask(Y_k):
    mask_k = get_same_shape_mask(Y_k)


    # Construct spatial mask
    m_vec = np.tile(mask_k, [1, Y_k.shape[0]])

    M = np.multiply(
        m_vec,
        np.eye(Y_k.shape[0])
    )

    return M



@jit
def kf_update_step(m_, P_, H_k, R_k, carry, x, innovation):
    """
    Computes the Kalman filter update equations with missing data support:
    
    In: 
        p(x_k | Y_{k-1}) = N(x_k | _m_k, _P_k)

    Computes:

        v_k = y_k - H_k _m_k 
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
    M = get_Y_mask(Y_k)
    Y_k = np.nan_to_num(Y_k)

    # -- KALMAN UPDATE --
    # m_, P_ is in latent - space -state format
    #mu = M @ H_k @ m_
    mu = M @ innovation
    var = M @ H_k @ P_ @ H_k.T @ M.T

    #inovation mean and variance
    # all in latent-space format
    v = Y_k - mu
    S = var + R_k

    K = solve(S, M @ H_k @ P_).T

    m_k = m_ + K @ v
    P_k = P_ - K @ S @ K.T

    #log marginal likelihood (assuming Gaussian likelihood)
    log_Z_k = np.sum(
        log_gaussian_with_mask(Y_k, mu, S, mask_k[:, 0])
    )


    if settings.kalman_filter_force_symmetric:
        P_k = force_symmetric(P_k)

    return {
        'm': m_k, 'P': P_k, 'm_inf': carry['m_inf'], 'P_inf': carry['P_inf']
    }, {
        'm': m_k, 'P': P_k, 'lml': log_Z_k
    }
@dispatch(UncertainPredictionInput, 'sequential', 'FITCSpatialSparsity')
def kf_predict_step(prior, carry, data, x, Xs_prior, lik_cov_flag):
    # dimensions of these should match
    base_prior = prior.prior
    # will pad non UI with Nones
    prediction_gp_list =  prior.prediction_gp

    # construct state-space form
    m_inf = carry['m_inf']
    P_inf =  carry['P_inf']
    dt_k = x['dt']
    X_s = data.X_space

    m_k = carry['m']
    P_k = carry['P']
    R_k =  x['lik_mat']

    H_k = prior.H(None, Xs_prior, None)

    m_, P_ = _kalman_predict(base_prior, m_k, P_k, Xs_prior, dt_k)

    kern_s = prior.base_prior.parent[0].kernel.k2

    # compute psi statistics
    scalar2mat = lambda a: np.array([a])[:, None]

    XS = scalar2mat(x['t'])
    X = Xs_prior
    pred_weights, pred_covar  = get_psi_statistics_linear_form_from_mu_var_from_gram(
        XS, 
        X,
        Y = H_k @ m_,
        K_ss = None,
        K_sx = None,
        K_xx = add_jitter(kern_s.K(X, X), settings.jitter),
        likelihood_var = np.zeros([Xs_prior.shape[0], Xs_prior.shape[0]]),
        noise_pred_mu = x['low_fidelity_prediction_mean'][:, None], 
        noise_pred_var = x['low_fidelity_prediction_covar'][:, None],
        base_gp_kernel = kern_s
    )

    # TODO: predictions will be funky


    H_k = pred_weights @ H_k

    # TODO: still quadractic here, even though we know that R_k is diagonal and pred_covar is diagonal
    pred_covar = np.diag(pred_covar[:, 0]) # Ns x Ns
    R_k = R_k + pred_covar

    innovation = H_k @ m_

    carry, state =  kf_update_step(m_, P_, H_k, R_k, carry, x, innovation)
    state['H'] = pred_weights
    return carry, state

@dispatch(LTI_SDE, 'sequential', 'FITCSpatialSparsity')
def kf_predict_step(prior, carry, data, x, Xs_prior, lik_cov_flag):
    # Compute innovation and R
    # pass flag to exploit woodbury?

    m_inf = carry['m_inf']
    P_inf =  carry['P_inf']

    H_k = prior.H(None, Xs_prior, None)

    dt_k = x['dt']
    X_s = data.X_space

    m_k = carry['m']
    P_k = carry['P']

    m_, P_ = _kalman_predict(prior, m_k, P_k, Xs_prior, dt_k)

    R_k =  x['lik_mat']

    # Compute sparsity transformation Kxz Kzz^{-1} m, diag(Kxz - Kxz Kzz^{-1} Kzx)
    kern_s = prior.base_prior.parent[0].kernel.k2

    # TODO: probably need temporal kernel variance somewhere here
    pred_weights, pred_covar = gaussian_prediction_diagonal_statistics(
        K_xs = kern_s.K_diag(X_s),
        K_xs_x = kern_s.K(X_s, Xs_prior),
        K_xx = add_jitter(kern_s.K(Xs_prior, Xs_prior), settings.jitter),
        lik_var = np.zeros([Xs_prior.shape[0], Xs_prior.shape[0]]) # sparsity is a noise free prediction
    )

    H_k = pred_weights @ H_k

    # TODO: still quadractic here, even though we know that R_k is diagonal and pred_covar is diagonal
    pred_covar = np.diag(pred_covar[:, 0]) # Ns x Ns
    R_k = R_k + pred_covar

    innovation = H_k @ m_

    return kf_update_step(m_, P_, H_k, R_k, carry, x, innovation)


@dispatch(LTI_SDE, 'sequential')
def kf_predict_step(prior, carry, data, x, X_s, lik_cov_flag):
    """ Linear Kalman Filter Predict Step """

    m_inf = carry['m_inf']
    P_inf =  carry['P_inf']

    H_k = prior.H(None, X_s, None)

    dt_k = x['dt']

    m_k = carry['m']
    P_k = carry['P']

    m_, P_ = _kalman_predict(prior, m_k, P_k, X_s, dt_k)

    innovation = H_k @ m_


    if lik_cov_flag:
        R_k =  x['lik_mat']
        return kf_update_step(m_, P_, H_k, R_k, carry, x, innovation)
    else:
        R_k_inv =  x['lik_mat']
        return kf_update_step_with_lik_precision(m_, P_, H_k, R_k_inv, carry, x)


@dispatch(SDE, 'sequential')
def kf_predict_step(model, carry, data, x, X_s, lik_cov_flag):
    """ Extended Kalman Filter Predict Step """
    H_k = model.H(None, X_s, None)

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


    if lik_cov_flag:
        R_k =  x['lik_mat']
        return kf_update_step(m_, P_, H_k, R_k, carry, x, H_k @ m_)
    else:
        R_k_inv =  x['lik_mat']
        return kf_update_step_with_lik_precision(m_, P_, H_k, R_k_inv, carry, x)

@dispatch(LinearizedFilter_SDE, 'sequential')
def kf_predict_step(prior, carry, data, x, X_s, lik_cov_flag):
    """ Form of Extended Kalman Filter Predict Step """

    P_inf = prior.P_inf(None, X_s, None)
    H_k = prior.H(None, X_s, None)

    dt_k = x['dt']

    m_k = carry['m']
    P_k = carry['P']

    A_k = prior.expm(X_s, dt_k)
    Q_k = prior.Q(dt_k, A_k, P_inf, X_spatial=X_s)


    m_ = A_k @ m_k
    P_ = A_k @ P_k @ A_k.T + Q_k

    #H_k is given by the cholesky of the 
    # Y is g(m)
    small_noise = 1e-6
    f = x['Y']
    H_jac_k = cholesky(x['lik_mat'])/(np.sqrt(small_noise))
    R_k = np.eye(m_.shape[0])*small_noise

    # construct a state dict for the pseudo observation update step
    x_psuedo = {
        'Y': np.zeros(f.shape[0])[:, None], 
        't': x['t'], 
        'dt': x['dt'], 
        'lik_mat': R_k, 
    }

    Ns_colocation = f.shape[0]
    return kf_update_step(m_, P_, H_jac_k, R_k, carry, x_psuedo, f)





@dispatch(PDE, 'sequential')
def kf_predict_step(model, carry, data, x, X_s, lik_cov_flag):
    """ Extended Kalman Filter Predict Step """

    if not lik_cov_flag:
        raise NotImplementedError()

    # model will be PDE[LTI_SDE[GP]]

    sde_prior = model.parent

    #P_inf = sde_prior.P_inf(None, X_s, None)

    F, L, Qc, _, _, P_inf = sde_prior.state_space_representation(X_s, None, None)
    H_sde_prior = sde_prior.H(None, X_s, None)

    dt_k = x['dt']
    m_k = carry['m']
    P_k = carry['P']

    A_k = sde_prior.expm(X_s, dt_k)

    if False:
        Q_k = lti_disc(F, Qc, L, x['dt'], settings.jitter, _state_block_dim(sde_prior, X_s))
    else:
        Q_k = sde_prior.Q(dt_k, A_k, P_inf, X_spatial=X_s)

    global_calibration = carry['global_calibration']

    # standard Kalman prediction
    m_ = A_k @ m_k
    P_ = A_k @ P_k @ A_k.T + Q_k

    if False:
        R_k =  x['lik_mat']
        innovation = H_sde_prior @ m_
        carry, ys =  kf_update_step(m_, P_,  H_sde_prior, R_k, carry, x, innovation)
        carry['global_calibration'] = global_calibration
        return carry, ys
    else:
        # full state
        #H_k = model.H(H_sde_prior@m_, X_s, x['t'])
        H_k = model.H(m_, X_s, x['t'])
        R_k =  x['lik_mat']

        if model.forcing_function is not None:
            force = x['forcing_function']

            # collocation method
            f = model.forward_g(H_sde_prior@m_, X_s, x['t'], force=force)
            H_jac_k = model.H_jac(H_sde_prior@m_, X_s, x['t'], force=force)

        else:
            # collocation method
            f = model.forward_g(m_, X_s, x['t'])
            H_jac_k = model.H_jac(m_, X_s, x['t'])

        if model.boundary_conditions is not None:
            # observer boundary conditions
            x_boundary = {
                'Y': x['boundary_data'], 
                't': x['t'], 
                'dt': x['dt'], 
                'lik_mat': x['lik_mat'], 
            }
            carry, ys = kf_update_step(m_, P_, H_jac_k @ H_sde_prior, R_k*0.0, carry, x_boundary, H_jac_k @ H_sde_prior @ m_)
            m_, P_ = carry['m'], carry['P']

        if True:
            # compute prediction with the PDE transform
            y_psuedo = model.psuedo_observations(X_s)
            #we only observer y_psuedo at the training locations, because we discretise the prior first
            # . then we obtain a Gaussian prior. Hence we should not observe y_psuedo at testing locations
            y_psuedo = y_psuedo * x['train_test_mask']

            # construct a state dict for the pseudo observation update step
            x_psuedo = {
                'Y': y_psuedo, 
                't': x['t'], 
                'dt': x['dt'], 
                'lik_mat': x['lik_mat'], 
            }

            Ns_colocation = f.shape[0]

            carry, ys = kf_update_step(m_, P_, H_jac_k , np.zeros((Ns_colocation, Ns_colocation)), carry, x_psuedo,  np.squeeze(f)[..., None])
            #carry, ys = kf_update_step(m_, P_,  H_jac_k @ H_sde_prior, np.eye(Ns_colocation)*1e-10, carry, x_psuedo, np.squeeze(f)[..., None])
            m_, P_ = carry['m'], carry['P']

            if True:
                # global calibration
                Y_k = y_psuedo
                f_k =  np.squeeze(f)[..., None]
                mask_k = get_same_shape_mask(Y_k)[:, 0]
                M = get_Y_mask(Y_k)
                Y_k = np.nan_to_num(Y_k)

                err = Y_k-f_k
                #HP_HT = H_jac_k @ H_sde_prior@P_ @ H_sde_prior.T@H_jac_k.T
                HP_HT = H_jac_k @P_ @H_jac_k.T

                if False:
                    # global error across whole state
                    mahal = mahal_with_mask(err, HP_HT, mask_k)
                    sigma_n = avg_mahal_with_mask(err, HP_HT, mask_k)
                    sigma_n = np.nan_to_num(sigma_n) # avoid nans due to degenerate P_, such as zero error at start

                if True:

                    # global error for each state dimension
                    # see https://arxiv.org/pdf/2012.08202
                    sigma_n = jax.vmap(lambda z, s: (z**2)/s, [0, 0])(err[:, 0], np.diag(HP_HT))
                    #sigma_n = err[1, 0]**2/P_[1, 1]
                    sigma_n = np.nan_to_num(sigma_n, posinf=0.0)
                    sigma_n = np.nan_to_num(sigma_n)

                # moving average
                global_calibration =  (global_calibration*(x['k']) + np.squeeze(sigma_n))/(x['k']+1)


        if model.observe_data:

            if model.observation_function is not None:
                _f = H_sde_prior@m_
                H_k = model.obs_jac(_f, X_s, x['t'])
                innovation = model.observation_function(_f, X_s, x['t'])
            else:
                #innovation = H_k @ H_sde_prior @ m_
                innovation = H_k  @ m_

            #carry, ys =  kf_update_step(m_, P_, H_k @ H_sde_prior, R_k, carry, x, innovation)
            carry, ys =  kf_update_step(m_, P_, H_k , R_k, carry, x, innovation)
            m_, P_ = carry['m'], carry['P']

    carry['global_calibration']  = global_calibration
    return carry, ys


def filter_step_wrapper(data, prior, Xs_prior, lik_cov_flag):
    sparsity_arr = prior.base_prior.get_sparsity()
    sparsity = sparsity_arr[0]

    # TODO: dispatch over everything
    if _ensure_str( sparsity) == 'NoSparsity':
        kf_predict_fn = evoke('kf_predict_step', prior, 'sequential')
    else:
        kf_predict_fn = evoke('kf_predict_step', prior, 'sequential', sparsity)

    def _fn(carry, x):
        return kf_predict_fn(prior, carry, data, x, Xs_prior, lik_cov_flag)

    return _fn

@dispatch('sequential')
def filter(data, prior, lik_mat, Y, X_t, Xs_prior, dt, lik_cov_flag, train_test_mask, train_index):
    """
    Args:
        lik_mat: is either R of R_inv, the block diagonal covariance ot the block_diagonal precision
    """

    # steady state does not depend on time
    # in latent-space-state format
    m_inf = prior.m_inf(None, Xs_prior, None)
    P_inf = prior.P_inf(None, Xs_prior, None)

    if settings.balance_state_space:
        A_0 = prior.expm(Xs_prior, dt[0])
        T, T_inf = get_ss_balance_transformation(A_0)  
        m_inf, P_inf = _transform_ss_init_params(T, T_inf, m_inf, P_inf)

    step_wrap = filter_step_wrapper(data, prior, Xs_prior, lik_cov_flag)
    unroll = 1

    state_dict = {
        'dt': dt,
        't': X_t,
        'Y': Y,
        'lik_mat': lik_mat,
        'train_test_mask': train_test_mask,
        'train_index': train_index,
        'k': np.arange(dt.shape[0]) # current timestep
    }
    
    carry_dict = {
        'm': m_inf,
        'P': P_inf,
        'm_inf': m_inf, 
        'P_inf': P_inf 
    }

    # TODO: use dispatch here?
    if isinstance(prior, PDE):

        if prior.boundary_conditions is not None:
            boundary_conditions = prior.boundary_conditions
            # ensure same shape as Y
            boundary_conditions = np.array(boundary_conditions)[train_index]
            state_dict['boundary_data'] = boundary_conditions

        if prior.forcing_function is not None:
            # ensure same shape as Y
            forcing_function = prior.forcing_function
            forcing_function = np.array(forcing_function)[train_index]
            state_dict['forcing_function'] = forcing_function

        if Xs_prior is None:
            carry_dict['global_calibration'] = np.zeros(m_inf.shape[0])
        else: 
            carry_dict['global_calibration'] = np.zeros(Xs_prior.shape[0])

    elif isinstance(prior, UncertainPredictionInput):
        # TODO: need to predict in ST format then propogate it through
        # TODO: use batch/loop
        def _collect_low_fidelity_prediction(gp, data):
            if gp is not None:
                pred_mu, pred_var = gp.predict_f(data.X)

            else:
                # construct a dummy reponse
                pred_mu = np.ones(data.N)*onp.NaN
                pred_var = np.ones(data.N)*onp.NaN

            return np.squeeze(pred_mu), np.squeeze(pred_var)

        res = [_collect_low_fidelity_prediction(gp, data) for gp in prior.prediction_gp]
        # Nt x Ns x P
        state_dict['low_fidelity_prediction_mean'] = np.array([np.squeeze(res[i][0]) for i in range(len(res))]).T
        state_dict['low_fidelity_prediction_covar'] = np.array([np.squeeze(res[i][1]) for i in range(len(res))]).T

    if False:
        print("DEBUGGING RUNNIGN KF WITH FOR LOOP")
        for i in range(dt.shape[0]):
            carry_dict, _ = step_wrap(carry_dict, {key: state_dict[key][i] for key in state_dict.keys()})
        exit()

    carry, ys = scan(
        jax.remat(step_wrap),
        carry_dict,
        state_dict,
        unroll = unroll
    )

    lml = np.sum(ys['lml'])

    filter_res = {'m': ys['m'], 'P': ys['P']}

    filter_res['meta'] = {}


    if isinstance(prior, PDE):
        filter_res['meta']['global_calibration'] = carry['global_calibration']
        filter_res['meta']['lml'] = lml

    elif isinstance(prior, UncertainPredictionInput):
        filter_res['meta']['H'] = ys['H']

    return lml, filter_res

def filter_loop(data: 'SequentialData', prior: 'Prior', R=None, R_inv = None, filter_type=False, train_test_mask=None, train_index=None):
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

    if train_test_mask is None:
        train_test_mask = np.ones(data.Nt)

    if train_index is None:
        train_index = np.arange(Y.shape[0])

    dt = np.hstack([np.zeros(1), dt])

    Nt = data.Nt
    Ns = data.Ns
    P = data.P

    # Fix Y shape
    # Y has shape Nt x P x Ns
    chex.assert_shape(Y, [Nt, P, Ns])
    # flatten but still in time - latent - space format
    Y = np.reshape(Y, [data.Nt, -1])

    # Ensure rank 2 at each time step
    Y = Y[..., None]

    if R_inv is not None:
        lik_cov_flag = False
        lik_mat = R_inv
    else:
        lik_cov_flag = True
        lik_mat = R

    # sequential, parallel, square_root_svm
    if settings.verbose:
        print(f'running {filter_type} kalman filter')

    # spatial points that the prior is defined over
    Xs_prior = _get_prior_spatial_points(data, prior)

    filter_fn = evoke('filter', filter_type)
    #filter_fn = evoke('filter', 'sequential')

    lml, filter_res =  filter_fn(data, prior, lik_mat, Y, X_t, Xs_prior, dt, lik_cov_flag, train_test_mask, train_index)


    return lml, filter_res

