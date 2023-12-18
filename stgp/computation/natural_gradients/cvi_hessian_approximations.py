"""
When computing natural gradients we need to compute dE[log P(Y|T(F))]/dS, which is not gurarenteed to ensure P.S.D updates.

Here we compute Gaus-Newton style approximations of this hessian.
"""
import jax
import jax.numpy as np
from jax import  grad, jit, jacfwd, vjp
import chex
import objax
from batchjax import batch_or_loop, BatchType
from functools import partial

from ... import settings
from ...utils.nan_utils import get_same_shape_mask 
from ..matrix_ops import cholesky, cholesky_solve, triangular_solve, vec_add_jitter, add_jitter, lower_triangle, vectorized_lower_triangular_cholesky, vectorized_lower_triangular, lower_triangular_cholesky, lower_triangle, to_block_diag
from ...utils.utils import vc_keep_vars, get_parameters, get_var_name_with_id, get_batch_type
from ..elbos.elbos import compute_expected_log_liklihood, compute_expected_log_liklihood_with_variational_params
from ...dispatch import dispatch, evoke
from ..parameter_transforms import psd_retraction_map
from ..integrals.samples import _process_samples
from ..integrals.approximators import mv_block_monte_carlo, mv_mean_field_block_monte_carlo, mv_block_monte_carlo_list
from ...data import Data, TemporallyGroupedData, MultiOutputTemporalData, TemporalData, SpatioTemporalData, SpatialTemporalInput

from ...dispatch import _ensure_str

# Types imports
from ...approximate_posteriors import ConjugateApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullConjugateGaussian, FullGaussianApproximatePosterior, DataLatentBlockDiagonalApproximatePosterior, ApproximatePosterior, DiagonalGaussianApproximatePosterior, MeanFieldConjugateGaussian
from ...sparsity import NoSparsity, FreeSparsity, Sparsity, SpatialSparsity

from .exponential_family_transforms import xi_to_theta, theta_to_lambda, xi_to_expectation, expectation_to_xi, lambda_to_theta, theta_to_xi, theta_to_lambda_diagonal, lambda_to_theta_diagonal, reparametise_cholesky_grad

from ...transforms import MultiOutput

def data_decomposes_across_time(data) -> bool:
    time_data_types = ['TemporallyGroupedData', 'SpatioTemporalData', 'TemporalData', 'MultiOutputTemporalData']

    data_type = _ensure_str(data)

    if settings.cvi_ng_exploit_space_time:
        return data_type in time_data_types
    else:
        return False

def create_new_single_time_data_of_same_type(data, X, Y):
    data_type = _ensure_str(data)
    """ Constructs new data of the same type with new data points X, Y. Does not sort. """
    if data_type == 'Data':
        return Data(X, Y)
    elif data_type == 'TemporallyGroupedData':
        return TemporallyGroupedData(X=X, Y=Y, sort=False)
    elif data_type == 'SpatioTemporalData':
        return SpatioTemporalData(
            X=SpatialTemporalInput(X_time=X[0, 0, :1], X_space = X[0, :, 1:], train=False), 
            Y=Y, 
            sort=False
        )
    elif data_type == 'MultiOutputTemporalData':
        return MultiOutputTemporalData(X=X[0], Y=Y, sort=False)
    elif data_type == 'TemporalData':
        return TemporalData(X=X[0], Y=Y, sort=False)

    raise RuntimeError(f'Data type {data_type} not supported')

def get_likelihood_hessian(model, m, S, laplace_log_lik=False):

    data = model.data
    if data_decomposes_across_time(data):
        X_st, Y_st = data.X_st, data.Y_st
        # Nt x Ns x P x 1
        T_f = jax.vmap(
            lambda m_t, S_t, x_t, y_t: compute_u_to_tf(model, m_t[None, ...], S_t[None, ...], data=create_new_single_time_data_of_same_type(data, X=x_t[None, ...], Y=y_t[None, ...]))
        )(m, S, X_st, Y_st)

        if not laplace_log_lik:
            raise NotImplementedError()
        else:
            # batch over Nt and Ns, only evaluate the conditional var on the indiviual P x 1 outputs
            Lambda = jax.vmap( jax.vmap( lambda f: model.likelihood.conditional_var(f) ))(T_f)

            # laplace approximation of the hessian
            neg_Lambda = -(1/Lambda)
            neg_Lambda = neg_Lambda[..., 0, 0] # will be [Nt x Ns x P x 1]

            # ensure rank 4
            if len(neg_Lambda.shape) == 3:
                # this is required due to an inconsistency in return dimensions
                #  when wrapping a likelihood in a ProductLikelihood
                neg_Lambda = neg_Lambda[..., None]

    else:
        q = model.approximate_posterior
        prior = model.prior
        Y = model.data.Y

        # N x P x 1
        T_f = compute_u_to_tf(model, m, S)

        # TODO: only works for exponential family likelihoods atm
        # [N x P x B]
        if not laplace_log_lik:
            hess = batch_or_loop(
                lambda y, t, lik: jax.vmap(lik.log_hessian_scalar)(y, t),
                [Y.T, T_f[..., 0].T, model.likelihood.likelihood_arr],
                [0, 0, 0],
                dim = len(model.likelihood.likelihood_arr),
                out_dim=1,
                batch_type = get_batch_type(model.likelihood.likelihood_arr)
            )
            neg_Lambda = np.array(hess)
            neg_Lambda = (neg_Lambda.T)[..., None]
        else:

            Lambda = jax.vmap(
                lambda f: model.likelihood.conditional_var(f[None, :]) # []
            )(
                T_f[..., 0] # [N x P]
            )

            Lambda = Lambda[:, :, 0, 0]
            # laplace approximation of the hessian
            neg_Lambda = -(1/Lambda)
            
        neg_Lambda = np.reshape(neg_Lambda, Y.shape)

        # Mask out entries corresponding to missing observations
        # These should just be ignored from the sums
        # N x P
        Y_mask = get_same_shape_mask(Y)
        chex.assert_equal(neg_Lambda.shape, Y_mask.shape)
        # N x P 
        neg_Lambda = neg_Lambda * Y_mask

        # N x P x 1
        neg_Lambda = neg_Lambda[..., None]

    return neg_Lambda

def compute_u_to_f(m, q_m, q_S, return_var_only = False, data = None):
    """
    Compute q(f) = E_p(f | u) [q(u | q_m, q_S)]

    Args:
        m: model
    """
    likelihood = m.likelihood
    prior = m.prior
    approximate_posterior = m.approximate_posterior
    inference = m.inference

    if data is None:
        data = m.data


    # compute the marginal q(f)
    q_f_mu, q_f_var = evoke('marginal', approximate_posterior, likelihood, prior, whiten=inference.whiten, debug=False)(
        data, q_m, q_S, approximate_posterior, likelihood, prior, inference.whiten
    )
    # If the model is Multioutput q_f_mu will be a list and each element of the list
    #   will have rank [3] and [4]

    if not(type(q_f_mu) is list):
        chex.assert_rank([q_f_mu, q_f_var], [3, 4])
        q_f_mu = [q_f_mu]
        q_f_var = [q_f_var]

    if return_var_only:
        return q_f_var

    return q_f_mu, q_f_var

def compute_f_to_tf(m, q_f_mu, q_f_var):
    """
    Computes E[T(F)] ~ T(E[F]) by the delta method

    Args:
        q_f_mu: Shape N X Q x B or [(N X Q x B)]

    Output:
        q_f_mu: Shape  N x P x B
    """
    data = m.data

    # transform through non linear part
    if type(m.prior) == MultiOutput:
        q_f_res = []

        # transform each output separately
        # assumes that each output is a single output
        for i, p in enumerate(m.prior.parent):
            t_p = _process_samples(q_f_mu[i], lambda x:x, p)
            #q_f_res.append(np.squeeze(t_p))
            q_f_res.append(t_p[..., 0, 0])

        # fix shapes
        q_f_res = np.array(q_f_res).T
        chex.assert_rank(q_f_res, 2)

        q_f_res = q_f_res[..., None]
    else:
        q_f_res = _process_samples(q_f_mu[0], lambda x:x, m.prior)

    chex.assert_rank(q_f_res, 3)

    return q_f_res

def compute_u_to_tf(model, q_mu_z, q_var_z, data=None):
    """ Helper function to compute the transformations of u to T(F) """
    # compute u -> f
    q_f_mu, q_f_var = compute_u_to_f(model, q_mu_z, q_var_z, data=data)

    # compute f -> T(f)
    T_f = compute_f_to_tf(model, q_f_mu, q_f_var)
    chex.assert_rank(T_f, 3)


    return T_f

def gauss_newton(u, S, model,  laplace_log_lik=False, prediction_samples=None):
    """
    Args:
        u: Nt x Ms x D
    """
    chex.assert_rank([u, S], [3, 4])

    q = model.approximate_posterior
    prior = model.prior
    Y = model.data.Y

    if settings.cvi_ng_batch:
        # only minibatch once so everything is computed with the same batch
        if model.data.minibatch:
            # TODO: minibatching only works when sparsity is used. Assert this.
            model.data.batch()
    else:
        #do not batch so we use the same batch as used in teh ELBO computation
        pass

    def J_u(m):

        # jacobian of u -> T(f(u))
        # [N x P x B x M x Q x B]
        # TODO: this is very memory intensive :( 
        # Would rewriting it as jacobian vector product help?

        if data_decomposes_across_time(model.data):
            data = model.data

            # Nt x Ns x P x 1
            neg_Lambda = get_likelihood_hessian(model, m, S, laplace_log_lik=laplace_log_lik)

            X_st, Y_st = data.X_st, data.Y_st

            # Nt x Ns x P x 1 x Ms x 1
            J_u_tf = jax.vmap(
                lambda m_t, S_t, x_t, y_t: jax.jacfwd(
                    lambda _m_t: compute_u_to_tf(model, _m_t[None, ...], S_t[None, ...], data=create_new_single_time_data_of_same_type(data, X=x_t[None, ...], Y=y_t[None, ...]))
                )(m_t)
            )(m, S, X_st, Y_st)

            # clean up shapes
            J_u_tf = J_u_tf[:, :, :, 0, :, :] # Nt x Ns x P x Ms x 1
            neg_Lambda = neg_Lambda[..., None] # Nt x Ns x P x 1 x 1

            # TODO: there is an inconsistency in how temporally grouped stores Y
            # it should be Nt x P x Ns to match the Spatio-temporal Case

            if _ensure_str(data) == 'TemporallyGroupedData':
                Y_st_mask = get_same_shape_mask(Y_st) # Nt x Ns x P 
            else:
                Y_st_mask = get_same_shape_mask(Y_st) # Nt x P x Ns 
                Y_st_mask = np.transpose(Y_st_mask, [0, 2, 1]) # Nt x Ns x P 

            # create masks for missing lieklihoods
            neg_Lambda = neg_Lambda * Y_st_mask[..., None, None]

            # Gauss Newton approximation
            # TODO: should probably write as a jax.vjp
            # Nt x Ns x P x Ms x Ms
            G_vec = jax.vmap( # batch over time
                jax.vmap( #batch over space
                    jax.vmap( #batch over outputs
                        lambda a, b: a @ b @ a.T
                    ) 
                )
            )(
                J_u_tf, neg_Lambda
            )


            # create masks for missing data
            Y_reshaped_for_G_mask = Y_st_mask[..., None, None] # Nt x Ns x P x 1 x1
            G_mask = np.tile( Y_reshaped_for_G_mask, [1, 1, 1, G_vec.shape[-2], G_vec.shape[-1]])
            chex.assert_equal(G_mask.shape, G_vec.shape)

            # remove missing data from the natural gradient sum
            G_vec_masked = G_mask * G_vec

            G = np.sum(G_vec_masked, [1, 2]) # Nt x Ms x Ms
            G = G[:, None, ...] # Nt x 1 x Ms x Ms

            if settings.verbose:
                print('ST GAUSS NEWTON')

            if data.minibatch:
                G = G*data.minibatch_scaling

        else:
            neg_Lambda = get_likelihood_hessian(model, m, S, laplace_log_lik=laplace_log_lik)
            u_tf_fn = lambda m: compute_u_to_tf(model, m, S)
            J_u_tf = jax.jacfwd(u_tf_fn)(m)

            # sum over B?
            G_vec = jax.vmap(
                # sum over N
                lambda Ju: jax.vmap(
                        # sum over P/Q
                        jax.vmap(
                            lambda a, b: a @ b @ a.T
                        )
                    )(
                        Ju[:, :, None, ...], 
                        neg_Lambda[..., None]
                    ),
                3
            )(J_u_tf)

            Y_mask = get_same_shape_mask(Y)
            Y_reshaped_for_G_mask = Y_mask[None, ...][..., None, None, None, None] 

            G_mask = np.tile( Y_reshaped_for_G_mask, [G_vec.shape[0], 1, 1, 1, G_vec.shape[4], G_vec.shape[5], G_vec.shape[6]])
            chex.assert_equal(G_mask.shape, G_vec.shape)

            # remove missing data from the natural gradient sum
            G_vec_masked = G_mask * G_vec

            G = np.sum(G_vec_masked, [1, 2, 3, 6])[:, None, ...]

            if model.data.minibatch:
                G = G*model.data.minibatch_scaling

            if settings.verbose:
                print('D GAUSS NEWTON')


        if settings.debug_mode:
            breakpoint()

        return G

    approx_hessian = 0.5 *  J_u(u)
    chex.assert_rank(approx_hessian, 4)

    return approx_hessian

def laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples, laplace_log_lik=True, delta_u = True):
    q = model.approximate_posterior


    if _ensure_str(q) == 'MeanFieldConjugateGaussian':
        # block diagonal

        approx_posteriors = q.approx_posteriors
        q_mu_z, q_var_z = batch_or_loop(
            lambda q: q.surrogate.posterior_blocks(),
            [approx_posteriors],
            [0],
            dim = len(approx_posteriors),
            out_dim=2,
            batch_type = get_batch_type(approx_posteriors)
        )

        # fix shapes
        Q, N, L, B = q_mu_z.shape
        q_mu_z = np.transpose(q_mu_z, [1, 0, 2, 3])
        q_mu_z = np.reshape(q_mu_z, [N, Q*L, B])

        q_var_z = np.transpose(q_var_z[:, :, 0, ...], [1, 0, 2, 3])
        #q_var_z = jax.vmap(to_block_diag)(q_var_z)
        #q_var_z = q_var_z[:, None, ...]
        
    else:
        # get parameters of q(u) in time-latent-space order
        q_mu_z, q_var_z = q.surrogate.posterior_blocks()
        chex.assert_rank([q_mu_z, q_var_z], [3, 4])

    # delta u
    if delta_u:
        approx_hessian = gauss_newton(q_mu_z, q_var_z , model, laplace_log_lik=laplace_log_lik, prediction_samples=prediction_samples)
    else:
        def wrapped_fn(s):
            return  gauss_newton(s, q_var_z , model, laplace_log_lik=laplace_log_lik, prediction_samples=prediction_samples)

        # sample u here
        approx_hessian = mv_block_monte_carlo(
            wrapped_fn, 
            q_mu_z, 
            q_var_z, 
            generator = model.inference.generator, 
            num_samples = prediction_samples
        )

    if _ensure_str(q) == 'MeanFieldConjugateGaussian':
        # fix shapes
        approx_hessian = jax.vmap(to_block_diag)(approx_hessian)
        approx_hessian = approx_hessian[:, None, ...]
    chex.assert_rank(approx_hessian, 4)
    return approx_hessian


def get_full_gaussian_hessian_approximation(model, beta, prediction_samples, enforce_psd_type):
    if enforce_psd_type == 'gauss_newton':
        approx_hessian =  laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples, laplace_log_lik=False, delta_u = False)
    elif enforce_psd_type == 'gauss_newton_delta_u':
        approx_hessian =  laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples, laplace_log_lik=False, delta_u = True)
    elif enforce_psd_type == 'laplace_gauss_newton':
        approx_hessian =  laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples, laplace_log_lik=True, delta_u = False)
    elif enforce_psd_type == 'laplace_gauss_newton_delta_u':
        approx_hessian =  laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples, laplace_log_lik=True, delta_u = True)
    else:
        raise RuntimeError()

    chex.assert_rank(approx_hessian, 4)
    return approx_hessian






