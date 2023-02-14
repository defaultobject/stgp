"""
When computing natural gradients we need to compute dE[log P(Y|F)]/dS, which is not gurarenteed to ensure P.S.D updates.

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
from ..matrix_ops import cholesky, cholesky_solve, triangular_solve, vec_add_jitter, add_jitter, lower_triangle, vectorized_lower_triangular_cholesky, vectorized_lower_triangular, lower_triangular_cholesky, lower_triangle
from ...utils.utils import vc_keep_vars, get_parameters, get_var_name_with_id, get_batch_type
from ..elbos.elbos import compute_expected_log_liklihood, compute_expected_log_liklihood_with_variational_params
from ...dispatch import dispatch, evoke
from ..parameter_transforms import psd_retraction_map
from ..integrals.samples import _process_samples
from ..integrals.approximators import mv_block_monte_carlo, mv_mean_field_block_monte_carlo, mv_block_monte_carlo_list

# Types imports
from ...approximate_posteriors import ConjugateApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullConjugateGaussian, FullGaussianApproximatePosterior, DataLatentBlockDiagonalApproximatePosterior, ApproximatePosterior, DiagonalGaussianApproximatePosterior, MeanFieldConjugateGaussian
from ...sparsity import NoSparsity, FreeSparsity, Sparsity, SpatialSparsity

from .exponential_family_transforms import xi_to_theta, theta_to_lambda, xi_to_expectation, expectation_to_xi, lambda_to_theta, theta_to_xi, theta_to_lambda_diagonal, lambda_to_theta_diagonal, reparametise_cholesky_grad

from ...transforms import MultiOutput

def compute_u_to_f(m, q_m, q_S, return_var_only = False):
    """
    Compute q(f) = E_p(f | u) [q(u | q_m, q_S)]

    Args:
        m: model
    """
    data = m.data
    likelihood = m.likelihood
    prior = m.prior
    approximate_posterior = m.approximate_posterior
    inference = m.inference


    N = data.N

    if data.minibatch:
        # TODO: minibatching only works when sparsity is used. Assert this.
        data.batch()

    # compute the marginal q(f)
    q_f_mu, q_f_var = evoke('marginal', approximate_posterior, likelihood, prior, whiten=inference.whiten)(
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

    if data.minibatch:
        # TODO: minibatching only works when sparsity is used. Assert this.
        data.batch()

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

def compute_u_to_tf(model, q_mu_z, q_var_z):
    """ Helper function to compute the transformations of u to T(F) """
    # compute u -> f
    q_f_mu, q_f_var = compute_u_to_f(model, q_mu_z, q_var_z)

    # compute f -> T(f)
    T_f = compute_f_to_tf(model, q_f_mu, q_f_var)
    chex.assert_rank(T_f, 3)

    return T_f

def gauss_newton_delta_f(u, S, model):
    chex.assert_rank([u, S], [3, 4])

    q = model.approximate_posterior
    prior = model.prior
    Y = model.data.Y

    if True:
        q_f_mu, q_f_var = compute_u_to_f(model, u, S)

        def J_f(f):
            T_f = compute_f_to_tf(model, f, None)

            f2tf = lambda *f: compute_f_to_tf(model, [fi[None, ...] for fi in f], None)[0]
            T_f = jax.vmap(f2tf)(*f)
            Q = len(f)

            # N x P x B x P x B
            J = jax.vmap(jax.jacfwd(f2tf, argnums=range(Q)))(*f)

            # N x Q x P 
            #J = J[:, :, 0, :, 0]

            # TODO: only works for Gaussian atm
            # Assuming that the likelihood components are (conditionally) indpedent
            # N x P x P
            # Approximating d^2 log P(Y | T) / dT^2 ~ - COV(Y | T)^{-1}
            neg_Lambda = jax.vmap(lambda f: np.diag(1/np.squeeze(model.likelihood.conditional_var(f[None, :]))))(T_f[..., 0])

            # Mask out entries corresponding to missing observations
            # These should just be ignored from the sums
            # N x P
            Y_mask = get_same_shape_mask(Y)
            Y_mask = np.tile(Y_mask[..., None], [1, 1, Y.shape[1]])
            neg_Lambda = neg_Lambda * Y_mask

            # Generalised Gauss-Newton
            #J_sum = (J**2) * neg_Lambda

            J_list = []
            for i in range(Q):
                J_i  = np.transpose(J[i][:, :, 0, :, 0], [0, 2, 1])
                J_vec = jax.vmap(lambda a, b, c: a @ b @ c.T)(J_i, neg_Lambda, J_i)
                J_list.append(J_vec[:, None, ...])

            return J_list

        if False:
            J_list = J_f(q_f_mu)
        else:
            J_list = mv_block_monte_carlo_list(
                J_f, 
                q_f_mu, 
                q_f_var, 
                generator = model.inference.generator, 
                num_samples = 100
            )
      
        S_f, vjp_fn = jax.vjp(lambda S: compute_u_to_f(model, u, S, return_var_only=True), S)
        var_grads = vjp_fn(J_list)[0]
        approx_hessian = - 0.5 * var_grads[:, 0, ...]
    else:

        # N x P x 1
        T_f = compute_u_to_tf(model, u, S)
        # compute jacobian of U_bar -> T(F_bar)
        # Use forward mode as it is more memory efficient
        # J will have shape (N x P x B) x (M x Q x B)
        # When in a state-space form M will be Mt and Q will be Ms
        J = jax.jacfwd(compute_u_to_tf, argnums=1)(model, u, S)


        # N x P x Mt x Ms x B
        J = J[:, :, 0, :, :, :]

        # TODO: check that B is handled correctly here, when B = 1 it does not matter
        # N x P x Mt x B x Ms
        J_T = np.transpose(J, [0, 1, 2, 4, 3])


        # TODO: only works for Gaussian atm
        # Assuming that the likelihood components are (conditionally) indpedent
        # N x P x 1 x 1
        cond_var = jax.vmap(lambda f: model.likelihood.conditional_var(f[None, ...]))(T_f)

        # N x P
        cond_var = cond_var[:, :, 0, 0]

        # Approximating d^2 log P(Y | T) / dT^2 ~ - COV(Y | T)^{-1}
        neg_Lambda =  1 / cond_var

        # Generalised Gauss-Newton
        # vmap over N and P
        J_sum = jax.vmap( jax.vmap(lambda a, b, c: a @ (b * c)))(J, neg_Lambda, J_T)

        # Mask out entries corresponding to missing observations
        # These should just be ignored from the sums
        # N x P
        Y_mask = get_same_shape_mask(Y)

        # Tile to same shape as J_sum
        J_mask = np.transpose(
            np.tile(Y_mask,  [J_sum.shape[2], J_sum.shape[3], J_sum.shape[4], 1, 1]), 
            [3, 4, 0, 1, 2]
        )
        chex.assert_equal_shape([J_mask, J_sum])

        # Mask
        J_sum = J_sum * J_mask

        # Sum over N and P
        var_grads = np.sum(J_sum, axis=[0, 1])

        # 0.5 comes from Barnett and Price 
        # negative sign comes from the laplace approximation of log likelihood hessian
        approx_hessian = - 0.5 * var_grads

    # fix shape
    approx_hessian = approx_hessian[:, None, ...]
    chex.assert_rank(approx_hessian, 4)

    return approx_hessian


def laplace_gauss_newton_delta_u_delta_f_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples):
    q = model.approximate_posterior


    # get parameters of q(u) in time-latent-space order
    q_mu_z, q_var_z = q.surrogate.posterior_blocks()
    chex.assert_rank([q_mu_z, q_var_z], [3, 4])

    # delta u
    approx_hessian = gauss_newton_delta_f(q_mu_z, q_var_z , model)
    chex.assert_rank(approx_hessian, 4)
    return approx_hessian

def laplace_gauss_newton_delta_f_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples):
    q = model.approximate_posterior
    prior = model.prior
    Y = model.data.Y

    # get parameters of q(u) in time-latent-space order
    q_mu_z, q_var_z = q.surrogate.posterior_blocks()
    chex.assert_rank([q_mu_z, q_var_z], [3, 4])


    # sample u here
    def wrapped_fn(s):
        return  gauss_newton_delta_f(s, q_var_z , model)

    approx_hessian = mv_block_monte_carlo(
        wrapped_fn, 
        q_mu_z, 
        q_var_z, 
        generator = model.inference.generator, 
        num_samples = prediction_samples
    )

    chex.assert_rank(approx_hessian, 4)
    return approx_hessian


def get_full_gaussian_hessian_approximation(model, beta, prediction_samples, enforce_psd_type):
    if enforce_psd_type == 'laplace_gauss_newton':
        approx_hessian =  laplace_gauss_newton_delta_u_delta_f_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples)
    elif enforce_psd_type == 'laplace_gauss_newton_delta_f':
        approx_hessian =  laplace_gauss_newton_delta_f_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples)
    else:
        raise RuntimeError()

    chex.assert_rank(approx_hessian, 4)
    return approx_hessian






