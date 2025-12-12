from ...dispatch import _ensure_str
import numpy as onp
import jax
import jax.numpy as np
from jax import jit
from ... import settings 
from ...utils.utils import is_empty_space
from ...utils.nan_utils import get_same_shape_mask
from ..linalg import solve, solve_from_cholesky
from ..gaussian import log_gaussian, log_gaussian_with_mask, log_gaussian_with_additive_precision_noise_with_mask, avg_mahal_with_mask, mahal_with_mask
from ..matrix_ops import cholesky, cholesky_solve, add_jitter, mat_inv, solve_with_additive_inverse, force_symmetric, lti_disc, to_block_diag
from ...computation.kernel_psi_statistics import get_fitc_sparsity_transformation, get_psi_statistics_linear_form_from_mu_var_from_gram



# Import types
from ...transforms.sdes import SDE, LTI_SDE
from ...transforms.pdes import PDE
from ...transforms.uncertain_inputs import UncertainPredictionInput


        
def padd_spatial_points_across_latents_with_zero(pred_weights, Xs_prior, q):
    """
    To handle multiple latent functions with varying dimenions we pad them out with zeros
        to make them the same dimension but without effecting the model.

    This function is to be used within a loop that is padding out each of the latent functions.

    inputs:
        pred_weights: the current latent function we wish to pad out
        Xs_prior: all spatial functions (ie to get the dimensions of all the latent functions)
        q: the index of the current latent function
    outputs:
        pred_weights_q: the padded out latent function

    """
    num_latents = len(Xs_prior)

    prior_num_spatial = [xs.shape[0] for xs in Xs_prior]
    prior_num_spatial_cumsum = onp.cumsum(prior_num_spatial).tolist()
    prior_num_spatial_cumsum_rev = list(reversed(onp.cumsum(list(reversed(prior_num_spatial))).tolist()))

    if q == 0:
        pred_weights_q = np.concatenate([
            pred_weights,
            np.zeros([*pred_weights.shape[:-1]]+[sum(prior_num_spatial_cumsum_rev[q+1:])])
        ], axis=-1)
    elif q == num_latents-1:
        pred_weights_q = np.concatenate([
            np.zeros([*pred_weights.shape[:-1], sum(prior_num_spatial_cumsum[:q])]), 
            pred_weights
        ], axis=-1)
    else:
        pred_weights_q = np.concatenate([
            np.zeros([*pred_weights.shape[:-1], sum(prior_num_spatial_cumsum[:q])]), 
            pred_weights,
            np.zeros([*pred_weights.shape[:-1], sum(prior_num_spatial_cumsum_rev[q+1:])])
        ], axis=-1)



    return pred_weights_q


def _get_data_or_sparsity_spatial_points(data, sparsity):
    """ Get spatial points of prior. If not using FITC then spatial points will be same as data """
    if _ensure_str( sparsity) == 'FITCSpatialSparsity':
        return sparsity.raw_Z.X_space
    else:
        return data.X_space

def _get_prior_spatial_points(data, prior):
    sparsity = prior.base_prior.get_sparsity()

    # TODO: this is a hack so we don't break all existing code to handle uncertain inputs
    # at some point refactor as it would be nice if we supported multiple latent functiosn 
    # with different spatial inducing point locations
    if  _ensure_str(prior) == 'UncertainPredictionInput' or _ensure_str(prior.parent) == 'UncertainPredictionInput':
        return [
            _get_data_or_sparsity_spatial_points(data, sparsity[i])
            for i in range(len(sparsity))
        ]
    else:
        return [_get_data_or_sparsity_spatial_points(data, sparsity) for sparsity in sparsity]

def _setup_pde_state_and_args(data, prior, m_inf, P_inf, Xs_prior, state_dict, args_dict, smoother=False, is_prediction=False):
    if smoother:
        return state_dict, args_dict
    else:
        # setup filter state
        train_index = args_dict['train_index']

        if prior.boundary_conditions is not None:
            boundary_conditions = prior.boundary_conditions
            # ensure same shape as Y
            boundary_conditions = np.array(boundary_conditions)[train_index]
            args_dict['boundary_data'] = boundary_conditions

        if prior.forcing_function is not None:
            # ensure same shape as Y
            forcing_function = prior.forcing_function
            forcing_function = np.array(forcing_function)[train_index]
            args_dict['forcing_function'] = forcing_function

        # global_calibration is part of state as it is something we will compute whilst filtering
        # check if there are actually any spatial points
        if Xs_prior is None or (type(Xs_prior) is list and all([xs is None for xs in Xs_prior])):
            state_dict['global_calibration'] = np.zeros(m_inf.shape[0])
        else: 
            # TODO: hmm, why is it the shape of the spatial points, not 
            # Ns x dt x ds? ie where is the state size?
            if type(Xs_prior) is list:
                xs_arr = [1 if xs is None else  xs.shape[0] for xs in Xs_prior]
                state_dict['global_calibration'] = np.zeros(sum(xs_arr))
            else:
                state_dict['global_calibration'] = np.zeros(Xs_prior.shape[0])

        return state_dict, args_dict

def _setup_uncertain_inputs_state_and_args(data, prior, m_inf, P_inf, Xs_prior, state_dict, args_dict, smoother=False, is_prediction=False):
    """
    Args:
        use_posterior: instead of using predict functions (which are not jittable) use the posterior 
    """
    # TODO: need to predict in ST format then propogate it through
    # TODO: use batch/loop
    def _collect_low_fidelity_prediction(gp, data):
        if gp is not None:
            if is_prediction:
                # only use the first dimension
                pred_mu, pred_var = gp.predict_f(data.X[:, [0]], force_full_state=True)
                pred_mu = pred_mu[..., 0]
                pred_var = pred_var[..., 0, 0]

                if data.X.shape[1] > 1 and settings.sde_ui_allow_certain_prediction:
                    try:
                        # predict at certain inputs for debugging
                        if False:
                            test_mask = 1-np.nan_to_num(args_dict['train_test_mask'])

                            test_mask = test_mask[:, None, None]

                            pred_mu = data.X[..., [1]] * test_mask + pred_mu * (1-test_mask)
                            pred_var = test_mask[..., None] *0.0 + pred_var * (1-test_mask[..., None])
                        if settings.filter_extra_debug_flag:
                            breakpoint()
                    except KeyError as e:
                        print(e)
                        print(args_dict.keys())
                        breakpoint()
                        pass
                if pred_mu.shape[1] > 1:
                    pred_mu = pred_mu[:, 0]
                    pred_var = pred_var[:, 0]

            else:
                pred_mu, pred_var = gp.posterior_in_latent_data()

                if pred_mu.shape[0] > 1:
                    pred_mu = pred_mu[0]
                    pred_var = pred_var[0]
        else:
            # construct a dummy reponse
            pred_mu = np.ones(data.N)*onp.nan
            pred_var = np.ones(data.N)*onp.nan

        return np.squeeze(pred_mu), np.squeeze(pred_var)

    res = [_collect_low_fidelity_prediction(gp, data) for gp in prior.prediction_gp]
    # Nt x Ns x P
    # TODO: generalize beyond P = 1 
    args_dict['low_fidelity_prediction_mean'] = np.array([np.squeeze(res[i][0]) for i in range(len(res))]).T
    args_dict['low_fidelity_prediction_covar'] = np.array([np.squeeze(res[i][1]) for i in range(len(res))]).T

    return state_dict, args_dict

def _setup_state_and_args_dict(data, prior, m_inf, P_inf, Xs_prior, state_dict, args_dict, smoother=False, is_prediction=False):
    # TODO: just assuming that the model has been constructed correctly
    # Loop through all priors and setup state and dict for each transformation

    # TODO: use dispatch here?
    if isinstance(prior, PDE):
        state_dict, args_dict = _setup_pde_state_and_args(
            data, prior, m_inf, P_inf, Xs_prior, state_dict, args_dict, smoother=smoother, is_prediction=is_prediction
        )

        return _setup_state_and_args_dict(data, prior.parent, m_inf, P_inf, Xs_prior, state_dict, args_dict, smoother=smoother, is_prediction=is_prediction)

    elif isinstance(prior, UncertainPredictionInput):
        state_dict, args_dict = _setup_uncertain_inputs_state_and_args(
            data, prior, m_inf, P_inf, Xs_prior, state_dict, args_dict, smoother=smoother, is_prediction=is_prediction
        )
        return _setup_state_and_args_dict(data, prior.parent, m_inf, P_inf, Xs_prior, state_dict, args_dict, smoother=smoother, is_prediction=is_prediction)

    # base prior is always LTI_SDE
    if isinstance(prior, LTI_SDE):
        return state_dict, args_dict

def _process_filter_results_state_and_args_dict(data, prior, m_inf, P_inf, Xs_prior, state, ys, filter_res, init_state_dict, init_args_dict):
    if not('meta' in filter_res.keys()):
        filter_res['meta'] = {}

    if isinstance(prior, PDE):
        filter_res['meta']['global_calibration'] = state['global_calibration']
        return _process_filter_results_state_and_args_dict(data, prior.parent, m_inf, P_inf, Xs_prior, state, ys, filter_res, init_state_dict, init_args_dict)

    elif isinstance(prior, UncertainPredictionInput):
        filter_res['meta']['H'] = ys['H']
        filter_res['meta']['ui_var'] = ys['ui_var']
        filter_res['meta']['low_fidelity_prediction_mean'] = init_args_dict['low_fidelity_prediction_mean']
        filter_res['meta']['low_fidelity_prediction_covar'] = init_args_dict['low_fidelity_prediction_covar']
        return _process_filter_results_state_and_args_dict(data, prior.parent, m_inf, P_inf, Xs_prior, state, ys, filter_res, init_state_dict, init_args_dict)

    # base prior is always LTI_SDE
    if isinstance(prior, LTI_SDE):
        return filter_res

def _construct_filter_with_pde_transform(m_, P_, R_k, H_sde_prior, x, carry, data, model, Xs_prior):
    sde_prior = model.parent
    # TODO: figure out where sde_prior should be used or not

    # full state
    #H_k = model.H(H_sde_prior@m_, X_s, x['t'])
    # TODO: this is essentially the parent H... Why do we need this if we have H_sde_prior?
    # Only used if data is observed, so probably not needed
    H_k = model.H(m_, Xs_prior, x['t'])

    global_calibration = carry['global_calibration']

    if model.forcing_function is not None:
        force = x['forcing_function']

        # collocation method
        f = model.forward_g(H_sde_prior @ m_, Xs_prior, x['t'], force=force)
        H_jac_k = model.H_jac(H_sde_prior @ m_, Xs_prior, x['t'], force=force)

    else:
        # collocation method
        f = model.forward_g(H_sde_prior @ m_, Xs_prior, x['t'])
        H_jac_k = model.H_jac(H_sde_prior @ m_, Xs_prior, x['t'])

    if model.boundary_conditions is not None:
        # observe boundary conditions
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
        y_psuedo = model.psuedo_observations(data.X_space)
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
        
        carry, ys = kf_update_step(m_, P_, H_jac_k @ H_sde_prior, np.zeros((Ns_colocation, Ns_colocation)), carry, x_psuedo,  f)
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
            HP_HT = H_jac_k @ H_sde_prior@P_ @ H_sde_prior.T@H_jac_k.T
            #HP_HT = H_jac_k @P_ @H_jac_k.T

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
            # hmm probably need to linearise here...
            if model.forcing_function is not None:
                H_k = model.obs_jac(_f, Xs_prior, x['t'], force=x['forcing_function'])
            else:
                H_k = model.obs_jac(_f, Xs_prior, x['t'])

            # TODO: is this correct? does it need to also be pushed into the kf_update_step

            if model.forcing_function is not None:
                innovation = model.observation_function(_f, Xs_prior, x['t'], force=x['forcing_function'])
            else:
                innovation = model.observation_function(_f, Xs_prior, x['t'])


            H_k = H_k @ H_sde_prior
        else:
            #innovation = H_k @ H_sde_prior @ m_
            #innovation = H_k  @ m_
            innovation = H_sde_prior  @ m_
            H_k =  H_sde_prior

        if model.observation_noise_function is not None:
            if model.forcing_function is not None:
                # jacobian wrt force
                force_jac = jax.jacfwd(
                    lambda force: model.observation_function(_f, Xs_prior, x['t'], force=force)
                )(x['forcing_function'])
                force_jac = force_jac[:, 0, :] # P x Q

                R_k = model.observation_noise_function(R_k, _f, Xs_prior, x['t'], force_jac=force_jac, force=x['forcing_function'])
            else:
                R_k = model.observation_noise_function(R_k, _f, Xs_prior, x['t'])

        # TODO: figure out H_k
        carry, ys =  kf_update_step(m_, P_, H_k , R_k, carry, x, innovation)
        m_, P_ = carry['m'], carry['P']

    carry['global_calibration']  = global_calibration

    return carry, ys

def get_Y_mask(Y_k):
    mask_k = get_same_shape_mask(Y_k)


    # Construct spatial mask
    m_vec = np.tile(mask_k, [1, Y_k.shape[0]])

    M = np.multiply(
        m_vec,
        np.eye(Y_k.shape[0])
    )

    return M

def uncertain_inputs_fitc_sparsity_compute_psi_statistics(prior, x, m_, P_, H_k_blocks, H_k_latents_only_blocks, Xs_prior):
    # compute psi statistics

    H_k_latents_only = to_block_diag(H_k_latents_only_blocks)

    # TODO: want this to be separate in filter_utils probably
    scalar2mat = lambda a: np.array([a])[:, None]

    XS = scalar2mat(x['t'])
    #for each latent function compute psi statistics 
    # TODO: rewrite using batching
    latents = prior.base_prior.parent
    num_latents = len(latents)

    pred_weights_arr = []
    pred_covar_arr = []
    for q in range(num_latents):
        X_q = Xs_prior[q]

        # check if there are spatial points
        # TODO: pull out into separate fn as it  reused?
        X_q_has_spatial_points = False
        # use onp since Xs_prior is not traced
        print('Xs_prior[q]: ', Xs_prior[q])
        if isinstance(Xs_prior[q], np.ndarray):
            if Xs_prior[q].shape[-1] != 0:
                X_q_has_spatial_points = True
        else:
            if Xs_prior[q] is not None:
                X_q_has_spatial_points = True

        if X_q_has_spatial_points:
            # TODO: does this need to be scaled by Kt?
            kern_s_q = latents[q].kernel.k2
            # compute psi statatics
            arr = []
            idx_slice_start = 0
            idx_slice = None

            # TODO: need to write out what this is actually doing
            for i in range(num_latents):
                # if  temporal model Xs_prior will be None, is a spatio-temporal model then number of spatial points coudl be zero
                X_i_has_spatial_points = False
                if isinstance(Xs_prior[i], np.ndarray):
                    if Xs_prior[i].shape[-1] != 0:
                        X_i_has_spatial_points = True
                else:
                    if Xs_prior[i] is not None:
                        X_i_has_spatial_points = True

                print(q, ' -- ', i, ' -- ', Xs_prior[i], X_i_has_spatial_points)

                if  X_i_has_spatial_points:
                    if i == q:
                        arr.append(np.eye(X_q.shape[0]))
                        idx_slice = slice(idx_slice_start, idx_slice_start+X_q.shape[0])
                    else:
                        arr.append(np.zeros((Xs_prior[i].shape[0], Xs_prior[i].shape[0])))
                        idx_slice_start += Xs_prior[i].shape[0]
                else:
                    arr.append(np.array([0]))
                    idx_slice_start += 1


            H_q = to_block_diag(arr)
            # only keep the parts relevenat to this latent function
            H_q = H_q[idx_slice]

            print(q, ' -- ', H_q.shape)

            pred_weights, pred_covar  = get_psi_statistics_linear_form_from_mu_var_from_gram(
                XS, 
                X_q,
                Y = H_q @ H_k_latents_only @ m_, # use H_k_latents_only as we only need the marginals not derivatives
                K_ss = None,
                K_sx = None,
                K_xx = add_jitter(kern_s_q.K(X_q, X_q), settings.jitter),
                likelihood_var = np.zeros([X_q.shape[0], X_q.shape[0]]),
                noise_pred_mu = x['low_fidelity_prediction_mean'][np.array([q])][:, None], 
                noise_pred_var = x['low_fidelity_prediction_covar'][np.array([q])][:, None],
                base_gp_kernel = kern_s_q
            )
            pred_weights_arr.append(pred_weights @ H_k_latents_only_blocks[q])
            pred_covar_arr.append(pred_covar)
            if settings.filter_extra_debug_flag:
                breakpoint()

        else:
            #no need to compute psi statistics
            # this should use the base H_q not just observe the latent
            # TODO: need H blocks then will just have access to it imediately...
            pred_weights_arr.append(H_k_blocks[q])
            pred_covar_arr.append(np.zeros([H_k_blocks[q].shape[0], H_k_blocks[q].shape[0]]))



    pred_weights = to_block_diag(pred_weights_arr)
    pred_covar = to_block_diag(pred_covar_arr)

    return pred_weights, pred_covar



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


