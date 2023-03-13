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
from ..permutations import data_order_to_output_order, permute_mat, permute_vec

from .cvi_hessian_approximations import get_full_gaussian_hessian_approximation
from .parameterisations import get_parameterisation_class 

# Types imports
from ...approximate_posteriors import ConjugateApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullConjugateGaussian, FullGaussianApproximatePosterior, DataLatentBlockDiagonalApproximatePosterior, ApproximatePosterior, DiagonalGaussianApproximatePosterior, MeanFieldConjugateGaussian
from ...sparsity import NoSparsity, FreeSparsity, Sparsity, SpatialSparsity

from .exponential_family_transforms import xi_to_theta, theta_to_lambda, xi_to_expectation, expectation_to_xi, lambda_to_theta, theta_to_xi, theta_to_lambda_diagonal, lambda_to_theta_diagonal, reparametise_cholesky_grad, lambda_to_theta_precision

from ...transforms import MultiOutput


@partial(jit, static_argnums=(7))
def cvi_diagonal_update(Y_tilde, V_tilde, m, s, m_grad, s_grad, beta, enforce_psd_type):
    # Get natural parameters for approximate likelihood
    lambda_1, lambda_2 = theta_to_lambda_diagonal(Y_tilde, V_tilde)

    # mu_grad and var_grad are ∂ell/∂θ 
    #calculate ∂ell/∂μ  = ∂ell/∂θ ∂θ/∂μ 
    grad_1 = m_grad - 2*s_grad*m
    grad_2 = s_grad

    # Natural gradient update updatae
    lambda_1_new  = (1-beta)*lambda_1 + beta* grad_1
    if enforce_psd_type == None:
        lambda_2_new  = (1-beta)*lambda_2 + beta* grad_2
    else:
        raise NotImplementedError()

    # Convert to theta
    theta_1, theta_2 = lambda_to_theta_diagonal(lambda_1_new, lambda_2_new)

    return theta_1, theta_2

@partial(jit, static_argnums=(7))
def cvi_block_update(Y_tilde, V_tilde, m, s, m_grad, s_grad, beta, enforce_psd_type):
    """
    In the conjugate setting the CVI update is:
        λ = λ + β (d/dμ) E[log p(Y | F)] 
    """
     # Ensure matrix input
    chex.assert_rank(
        [Y_tilde, V_tilde, m, s, m_grad, s_grad],
        [2, 2, 2, 2, 2, 2]
    )

    # Get natural parameters for approximate likelihood
    lambda_1, lambda_2 = theta_to_lambda(Y_tilde, V_tilde)

    # mu_grad and var_grad are ∂ell/∂θ 
    #calculate ∂ell/∂μ  = ∂ell/∂θ ∂θ/∂μ 
    grad_1 = m_grad - 2*s_grad @ m
    grad_2 = s_grad

    # Natural gradient update updatae
    lambda_1_new  = (1-beta)*lambda_1 + beta* grad_1

    if enforce_psd_type == None:
        lambda_2_new  = (1-beta)*lambda_2 + beta* grad_2
    elif enforce_psd_type == 'retraction':
        lambda_2_new = psd_retraction_map(-2*(1-beta)*lambda_2, -2*beta*grad_2)/(-2)
        #lambda_2_new = m_grad @ m_grad.T
        #lambda_2_new = (1-beta)*lambda_2 -  beta * np.sqrt((grad_1 @ grad_1.T))
    elif enforce_psd_type == 'riemannian':
        #breakpoint()
        lambda_2_new  = (1-beta)*lambda_2 + beta* grad_2
        #lambda_2_new = -np.abs(lambda_2_new)
        lambda_2_new = np.clip(lambda_2_new, a_max = -1e-5)
        #TODO
        #raise NotImplementedError()
    else:
        raise NotImplementedError()
    #lambda_2_new = np.clip(lambda_2_new, a_max = -1e-5)

    # Convert to theta
    #theta_1, theta_2 = lambda_to_theta(lambda_1_new, lambda_2_new)

    return lambda_1_new, lambda_2_new


@jit
def reparametise_vec_grad(m, m_grad, prior):
    #Calculate ∂L/μ = ∂L/∂ξ ∂ξ/μ 
    x, u = vjp(
        lambda v: prior.unpermute_vec(v), 
        m
    )
    m_grad = u(m_grad)[0]

    return m_grad


def _get_mf_params(model, diagonal=True):
    """ Helper function to wrap up batching over the latent GPs to collect variational parameters"""
    q = model.approximate_posterior

    # Get natural parameters
    q_list = model.approximate_posterior.approx_posteriors
    Q = len(q_list)

    # Collect CVI parameters
    raw_Y, Y_tilde_arr, V_tilde_arr = batch_or_loop(
        lambda q: (q.surrogate.data.base._Y.value, q.surrogate.data.base.Y, q.surrogate.likelihood.likelihood_arr[0].base.variance),
        [q_list],
        [0],
        dim=len(q_list),
        out_dim=2,
        batch_type = get_batch_type(q_list)
    )

    if diagonal:
        fn = lambda q: q.surrogate.posterior(diagonal=True)
    else:
        fn = lambda q: q.surrogate.posterior_blocks()

    # Compute q_m_z, q_S_z
    q_mu_z, q_var_z = batch_or_loop(
        fn,
        [q_list],
        [0],
        dim = Q,
        out_dim=2,
        batch_type = get_batch_type(q_list)
    )

    # each component returns rank [3, 4]. Remove the extra dimension and fix ordering.
    q_mu_z = np.transpose(q_mu_z[..., 0], [1, 0, 2])
    q_var_z = np.transpose(q_var_z[:, :, 0, ...], [1, 0, 2, 3])

    return raw_Y, Y_tilde_arr, V_tilde_arr, q_mu_z, q_var_z

def _get_marginals(model):
    q_m, q_S = evoke('variational_params', model.approximate_posterior, model.likelihood, model.prior.base_prior, model.inference.whiten)(
        model.data, model.approximate_posterior, model.likelihood, model.prior, model.inference.whiten
    )

    q_f_mu, q_f_var = evoke('marginal', model.approximate_posterior, model.likelihood, model.prior, whiten=model.inference.whiten)(
        model.data, q_m, q_S, model.approximate_posterior, model.likelihood, model.prior, model.inference.whiten
    )
    return q_f_mu, q_f_var 

def partial_ell(m, q_m, q_S):
    """ Helper function to compute the expected log likelihood using the variational paramters q_m, q_S"""

    return compute_expected_log_liklihood_with_variational_params(
        m.data,
        q_m,
        q_S,
        m.likelihood, 
        m.prior,
        m.approximate_posterior,
        m.inference
    )


@dispatch('VGP', MeanFieldConjugateGaussian, NoSparsity)
def natural_gradients(model, beta: float, enforce_psd_type) -> np.ndarray:
    raw_Y_arr, Y_tilde_arr, V_tilde_arr, q_mu_z, q_var_z = _get_mf_params(model, diagonal=False)
    breakpoint()


    # Different models store Y with different dimensions so we store it here so can 
    #   match the shape in the output
    Y_shape = raw_Y_arr.shape

    Q, N, B, _ = V_tilde_arr.shape

    # Fix shapes for ELL
    #breakpoint()
    #q_mu_z = np.reshape(q_mu_z, [Q, N*B, 1])
    #q_var_z = np.reshape(q_var_z, [Q, N*B, 1])

    # TODO --
    # We haev no sparsity and so q_mu, q_var should be diagonal/marginals
    #   however to keep code relatively simply we use the same block-diagonal likelihood as the sparsity case
    #   so to keep shapes consistent we extract the block diagonals here 
    # Compute dELL/dm, dEll/dS
    mu_grads, var_grads = jax.grad(partial_ell, (1, 2))(
        model, q_mu_z, q_var_z
    )


    # Fix shapes for Natgrads
    Y_tilde_arr = np.reshape(Y_tilde_arr, [Q, N, B, 1])
    V_tilde_arr = np.reshape(V_tilde_arr, [Q, N, B, B])

    q_mu_z = np.reshape(q_mu_z, [Q, N, B, 1])
    q_var_z = np.reshape(q_var_z, [Q, N, B, B])

    mu_grads = np.reshape(mu_grads, [Q, N, B, 1])
    var_grads = np.reshape(var_grads, [Q, N, B, B])

    # vmap over Q and N
    new_Y_tilde, new_V_tilde = jax.vmap(
        jax.vmap(cvi_block_update, [0, 0, 0, 0, 0, 0, None, None]),
        [0, 0, 0, 0, 0, 0, None, None]
    )(
        Y_tilde_arr, V_tilde_arr, q_mu_z, q_var_z, mu_grads, var_grads, beta, enforce_psd_type
    )

    # Fix shapes for output
    new_Y_tilde = np.reshape(new_Y_tilde, Y_shape)

    return new_Y_tilde, new_V_tilde

@dispatch('VGP', MeanFieldConjugateGaussian, SpatialSparsity)
def natural_gradients(model, beta: float, enforce_psd_type) -> np.ndarray:
    raw_Y_arr, Y_tilde_arr, V_tilde_arr, q_mu_z, q_var_z = _get_mf_params(model, diagonal=False)

    # Different models store Y with different dimensions so we store it here so can 
    #   match the shape in the output
    Y_shape = raw_Y_arr.shape

    Q, Nt, B, _ = V_tilde_arr.shape
    N = Nt*B

    # Fix shapes for ELL
    q_mu_z = np.reshape(q_mu_z, [Q, Nt, B])
    q_var_z = np.reshape(q_var_z, [Q, Nt, B, B])

    # Compute dELL/dm, dEll/dS
    mu_grads, var_grads = jax.grad(partial_ell, (1, 2))(
        model, q_mu_z, q_var_z
    )

    # Fix shapes for Natgrads
    Y_tilde_arr = np.reshape(Y_tilde_arr, [Q, Nt, B, 1])
    V_tilde_arr = np.reshape(V_tilde_arr, [Q, Nt, B, B])

    q_mu_z = np.reshape(q_mu_z, [Q, Nt, B, 1])
    q_var_z = np.reshape(q_var_z, [Q, Nt, B, B])

    mu_grads = np.reshape(mu_grads, [Q, Nt, B, 1])
    var_grads = np.reshape(var_grads, [Q, Nt, B, B])

    # vmap over Q and N
    new_Y_tilde, new_V_tilde = jax.vmap(
        jax.vmap(cvi_block_update, [0, 0, 0, 0, 0, 0, None, None]),
        [0, 0, 0, 0, 0, 0, None, None]
    )(
        Y_tilde_arr, V_tilde_arr, q_mu_z, q_var_z, mu_grads, var_grads, beta, enforce_psd_type
    )

    # Fix shapes for output
    new_Y_tilde = np.reshape(new_Y_tilde, Y_shape)

    return new_Y_tilde, new_V_tilde


@dispatch('VGP', MeanFieldConjugateGaussian, FreeSparsity)
def natural_gradients(model, beta: float, enforce_psd_type) -> np.ndarray:
    """
    Block CVI Natural Gradients
    """
    breakpoint()
    prior = model.prior

    # Get natural parameters
    q_list = model.approximate_posterior.approx_posteriors
    Q = len(q_list)

    # Collect CVI parameters
    Y_tilde_arr, V_tilde_arr = batch_or_loop(
        lambda q: (q.surrogate.Y, q.surrogate.likelihood.likelihood_arr[0].variance[0]),
        [q_list],
        [0],
        dim=len(q_list),
        out_dim=2,
        batch_type = get_batch_type(q_list)
    )

    Z_tiled = prior.get_Z()

    # Compute q_m_z, q_S_z
    mu_arr, var_arr = batch_or_loop(
        lambda q, z: q.surrogate.predict_f(z, diagonal=False),
        [q_list, Z_tiled],
        [0, 0],
        dim = Q,
        out_dim=2,
        batch_type = get_batch_type(q_list)
    )

    mu_arr = mu_arr[..., None]

    def partial_ell(m, q_m, q_S):
        q = MeanFieldApproximatePosterior(approximate_posteriors=[
            GaussianApproximatePosterior(m=q_m[q], S_inv=q_S[q], train=False)
            for q in range(q_m.shape[0])
        ])

        return compute_expected_log_liklihood(
            m.X, 
            m.Y, 
            m.likelihood, 
            m.prior,
            q,
            m.inference
        )

    mu_grads, var_chol_grads = jax.grad(partial_ell, (1, 2))(model, mu_arr, vectorized_lower_triangular_cholesky(vec_add_jitter(var_arr, settings.ng_jitter)))
    var_chol_grads = vectorized_lower_triangular(var_chol_grads, N=mu_grads[0].shape[0])

    # Reparameterise from cholesky grad to full matrix grad
    s_grads = jax.vmap(
        lambda var, var_chol: reparametise_cholesky_grad(var, var_chol, None, False),
        [0, 0],
        0
    )(var_arr, var_chol_grads)

    # Make sure shapes are correct
    chex.assert_shape(Y_tilde_arr, mu_grads.shape)
    chex.assert_shape(Y_tilde_arr, mu_arr.shape)
    chex.assert_shape(V_tilde_arr, s_grads.shape)
    chex.assert_shape(V_tilde_arr, var_arr.shape)

    # Update natural parameters
    new_Y_tilde, new_V_tilde = jax.vmap(
        cvi_block_update,
        [0, 0, 0, 0, 0, 0, None],
        0
    )(
        Y_tilde_arr, 
        V_tilde_arr, 
        mu_arr, 
        var_arr,
        mu_grads, 
        s_grads, 
        beta
    )

    chex.assert_shape(Y_tilde_arr, new_Y_tilde.shape)
    chex.assert_shape(V_tilde_arr, new_V_tilde.shape)

    new_V_tilde = new_V_tilde[:, None, ...]

    return new_Y_tilde, new_V_tilde


@dispatch('VGP', FullConjugateGaussian, FreeSparsity)
def natural_gradients(model, beta: float, enforce_psd_type) -> np.ndarray:
    q = model.approximate_posterior
    prior = model.prior
    sparsity_arr = prior.get_sparsity_list()
    surrogate_prior = q.surrogate.prior

    # Collect CVI parameters
    Y_tilde_arr, V_tilde_arr = q.surrogate.Y, q.surrogate.likelihood.variance

    Z_tiled = prior.get_Z()
    Z = np.vstack(Z_tiled)

    M = Z_tiled[0].shape[0]
    Q = Z_tiled.shape[0]

    # Data-latent order
    Y_tilde_arr = np.reshape(Y_tilde_arr, [-1, 1])
    V_tilde_arr = V_tilde_arr[0]

   # Predict in data-latent order
    q_mu_z, q_var_z = q.surrogate.predict_blocks(Z_tiled, M, M*Q, diagonal=False)
    q_mu_z, q_var_z = q_mu_z[0][..., None], q_var_z[0]


    # FullGaussianApproximatePosterior is meant to be used in latent-data order
    # So convert from data-latent order to latent-data
    q_mu_z_permuted = prior.unpermute_vec(q_mu_z)
    q_var_z_permuted = prior.unpermute_mat(q_var_z)

    def partial_ell(m, q_m, q_S):
        q = FullGaussianApproximatePosterior(
            m=q_m, S_inv=q_S, train=False
        )

        return compute_expected_log_liklihood(
            m.X, 
            m.Y, 
            m.likelihood, 
            m.prior,
            q,
            m.inference
        )

    mu_grads, var_chol_grads = jax.grad(partial_ell, (1, 2))(
        model, q_mu_z_permuted, lower_triangular_cholesky(add_jitter(q_var_z_permuted, settings.ng_jitter))
    )
    var_chol_grads = lower_triangle(var_chol_grads, N=mu_grads.shape[0])

    # Reparameterise gradients
    s_grad_permuted = reparametise_cholesky_grad(q_var_z, var_chol_grads, surrogate_prior, True)
    mu_grads_permuted = reparametise_vec_grad(q_mu_z, mu_grads, surrogate_prior)

    new_Y_tilde, new_V_tilde = cvi_block_update(
        Y_tilde_arr, V_tilde_arr, q_mu_z, q_var_z, mu_grads_permuted, s_grad_permuted, beta, enforce_psd_type
    )
    # Fix dimensions and order
    new_Y_tilde = np.reshape(new_Y_tilde, q.surrogate.Y.shape)
    new_V_tilde = new_V_tilde[None, ...]


    return new_Y_tilde, new_V_tilde

def partial_ell(m, q_m, q_S):
    # this is expecting q_m, q_S to be in time-latent-space format
    return compute_expected_log_liklihood_with_variational_params(
        m.data,
        q_m,
        q_S,
        m.likelihood, 
        m.prior,
        m.approximate_posterior,
        m.inference
    )

@dispatch('VGP', FullConjugateGaussian, NoSparsity)
@dispatch('VGP', FullConjugateGaussian, SpatialSparsity)
def natural_gradients(model, beta: float, enforce_psd_type) -> np.ndarray:
    """
    In the conjugate setting the CVI update is:
        λ = λ + β (d/dμ) E[log p(Y | F)] 

    There are three different situtations that we consider:
        - No Sparsity:  λ is block diagonal with blocks of size Q
        - Spatial Sparsity:  λ is block diagonal with blocks of size Ns * Q
        - Free Sparsity:  λ is a full matrix of size (N * Q) x (N * Q)
    """
    q = model.approximate_posterior
    prior = model.prior
    sparsity_arr = prior.base_prior.get_sparsity_list()

    # Collect CVI parameters
    # time-latent-space format
    #Y_tilde_arr is in time-space-latent format
    raw_Y_arr, Y_tilde_arr, V_tilde_arr = q.surrogate.data._Y.value, q.surrogate.Y, q.surrogate.likelihood.variance


    Nt, Nl, Ns = raw_Y_arr.shape

    # Different models store Y with different dimensions so we store it here so can 
    #   match the shape in the output
    Y_shape = raw_Y_arr.shape

    # Predict in time-latent-space order
    q_mu_z, q_var_z = q.surrogate.posterior_blocks()
    chex.assert_rank([q_mu_z, q_var_z], [3, 4])

    # Ensure correct size
    # still in time-latent-space format as reshape does not affect this
    N, Q = q_mu_z.shape[0], q_mu_z.shape[1]
    #q_mu_z = np.reshape(q_mu_z, [N, Q])
    #q_var_z = np.reshape(q_var_z, [N, Q, Q])

    # still in time-latent-space
    mu_grads, var_test = jax.grad(partial_ell, (1, 2))(
        model, q_mu_z, q_var_z
    )

    if enforce_psd_type in ['laplace_gauss_newton', 'laplace_gauss_newton--single', 'laplace_gauss_newton_delta_f']:
        var_grads = get_full_gaussian_hessian_approximation(model, beta, settings.ng_samples, enforce_psd_type)
        enforce_psd_type = None
    else:
        # in time-latent-space 
        var_grads = var_test

    # grads should be same as q_mu_z, q_var_z
    chex.assert_shape([mu_grads, var_grads], [q_mu_z.shape, q_var_z.shape])

    # Fix shapes
    # raw_Y_arr is in time-latent-space format, this reshape will preserve that
    Y_tilde_arr = np.reshape(raw_Y_arr, q_mu_z.shape)

    # update for each N
    new_Y_tilde, new_V_tilde = jax.vmap(
        cvi_block_update,
        [0, 0, 0, 0, 0, 0, None, None]
    )(
            Y_tilde_arr, V_tilde_arr, q_mu_z, q_var_z[:, 0, ...], mu_grads, var_grads[:, 0, ...], beta, enforce_psd_type
    )

    # reshape will preserve the data-latent format

    # convert back  to data-latent format
    #H = data_order_to_output_order(Nl, Ns).T
    #new_Y_tilde = jax.vmap(lambda a: H.T @ a)(new_Y_tilde)
    new_Y_tilde = np.reshape(new_Y_tilde, Y_shape)

    return new_Y_tilde, new_V_tilde

# =================== Different Parameterisations Entry Points ====================
# TODO: refactor to remove duplicated code
@dispatch('VGP', MeanFieldConjugateGaussian, "NG_Moment")
def natural_gradients(model, parameterisation, beta: float, enforce_psd_type, prediction_samples: int = None) -> np.ndarray:
    q = model.approximate_posterior
    prior = model.prior
    sparsity_arr = prior.base_prior.get_sparsity_list()
    parameterisation = get_parameterisation_class(q)

    # compute natural gradient update in natural parameterisation
    # TODO: assuming sparsity is constant across all latents
    lambda_1, lambda_2 = evoke('natural_gradients', model, q, sparsity_arr[0])(
        model, beta, enforce_psd_type
    )
    # convert to mean and covariance
    # vmap over P and blocks
    theta_1, theta_2 = jax.vmap(jax.vmap(lambda_to_theta))(lambda_1, lambda_2)
    return theta_1, theta_2

@dispatch('VGP', MeanFieldConjugateGaussian, "NG_Precision")
def natural_gradients(model, parameterisation, beta: float, enforce_psd_type, prediction_samples: int = None) -> np.ndarray:
    q = model.approximate_posterior
    prior = model.prior
    sparsity_arr = prior.base_prior.get_sparsity_list()
    parameterisation = get_parameterisation_class(q)

    # compute natural gradient update in natural parameterisation
    # TODO: assuming sparsity is constant across all latents
    lambda_1, lambda_2 = evoke('natural_gradients', model, q, sparsity_arr[0])(
        model, beta, enforce_psd_type
    )

    # convert to mean and precision
    # vmap over P and blocks
    theta_1, theta_2 = jax.vmap(jax.vmap(lambda_to_theta_precision))(lambda_1, lambda_2)
    return theta_1, theta_2

@dispatch('VGP', FullConjugateGaussian, "NG_Moment")
def natural_gradients(model, parameterisation, beta: float, enforce_psd_type, prediction_samples: int = None) -> np.ndarray:
    breakpoint()

@dispatch('VGP', FullConjugateGaussian, "NG_Precision")
def natural_gradients(model, parameterisation, beta: float, enforce_psd_type, prediction_samples: int = None) -> np.ndarray:
    breakpoint()

# =================== NG Entry Points ====================

@dispatch('VGP', ApproximatePosterior)
def natural_gradients(model, beta: float, enforce_psd_type, prediction_samples: int = None) -> np.ndarray:
    q = model.approximate_posterior
    parameterisation = get_parameterisation_class(q)

    return evoke('natural_gradients', model, q, parameterisation)(
        model, parameterisation, beta, enforce_psd_type, prediction_samples
    )
