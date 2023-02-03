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
from ..integrals.approximators import mv_block_monte_carlo

# Types imports
from ...approximate_posteriors import ConjugateApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullConjugateGaussian, FullGaussianApproximatePosterior, DataLatentBlockDiagonalApproximatePosterior, ApproximatePosterior, DiagonalGaussianApproximatePosterior, MeanFieldConjugateGaussian
from ...sparsity import NoSparsity, FreeSparsity, Sparsity, SpatialSparsity

from .exponential_family_transforms import xi_to_theta, theta_to_lambda, xi_to_expectation, expectation_to_xi, lambda_to_theta, theta_to_xi, theta_to_lambda_diagonal, lambda_to_theta_diagonal, reparametise_cholesky_grad

from ...transforms import MultiOutput

def compute_u_to_f(m, q_m, q_S):
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
            q_f_res.append(np.squeeze(t_p))

        # fix shapes
        q_f_res = np.array(q_f_res).T
        chex.assert_rank(q_f_res, 2)

        q_f_res = q_f_res[..., None]
    else:
        t_p = _process_samples(q_f_mu, lambda x:x, m.prior)
        chex.assert_rank(t_p, 3)
        q_f_res = t_p

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

def laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples):
    q = model.approximate_posterior
    prior = model.prior
    Y = model.data.Y

    # get parameters of q(u) in time-latent-space order
    q_mu_z, q_var_z = q.surrogate.posterior_blocks()
    chex.assert_rank([q_mu_z, q_var_z], [3, 4])

    # N x P x 1
    T_f = compute_u_to_tf(model, q_mu_z, q_var_z)
    # compute jacobian of U_bar -> T(F_bar)
    # Use forward mode as it is more memory efficient
    # J will have shape (N x P x B) x (M x Q x B)
    # When in a state-space form M will be Mt and Q will be Ms
    J = jax.jacfwd(compute_u_to_tf, argnums=1)(model, q_mu_z, q_var_z)

    # N x P x Mt x Ms x B
    J = J[:, :, 0, :, :, :]

    # TODO: check that B is handled correctly here, when B = 1 it does not matter
    # N x P x Mt x B x Ms
    J_T = np.transpose(J, [0, 1, 2, 4, 3])

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
    # negative sin comes from the laplace approximation of log likelihood hessian
    approx_hessian = - 0.5 * var_grads

    # fix shape
    approx_hessian = approx_hessian[:, None, ...]
    chex.assert_rank(approx_hessian, 4)

    return approx_hessian

def get_full_gaussian_hessian_approximation(model, beta, prediction_samples, enforce_psd_type):
    if enforce_psd_type == 'laplace_gauss_newton':
        approx_hessian =  laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples)
        return approx_hessian




