from ... import settings
from ..matrix_ops import cholesky, cholesky_solve, triangular_solve, add_jitter, lower_triangle, vectorized_lower_triangular_cholesky, vectorized_lower_triangular, lower_triangular_cholesky, lower_triangle
from ...utils.utils import vc_keep_vars, get_parameters, get_var_name_with_id, get_batch_type
from ...dispatch import dispatch, evoke
from ...approximate_posteriors import ConjugateApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullConjugateGaussian, FullGaussianApproximatePosterior

from ..elbos.prior_ops import prior_Z

from ..elbos.elbos import compute_expected_log_liklihood

import chex
from batchjax import batch_or_loop, BatchType
import jax
import jax.numpy as np
from jax import grad, jit, jacfwd
from jax import vjp

import objax

from typing import List


@jit
def xi_to_theta(xi1, xi2):
    return xi1, xi2 @ xi2.T

@jit
def theta_to_xi(theta_1, theta_2):
    M = theta_1.shape[0]
    jit = settings.ng_jitter * np.eye(M) 

    xi1 = theta_1
    xi2 = cholesky(theta_2+jit)

    return xi1, xi2

@jit
def theta_to_lambda(theta_1, theta_2):
    M = theta_1.shape[0]
    jit = settings.ng_jitter * np.eye(M) 

    theta_2_chol = cholesky(theta_2+jit)

    theta_2_chol_inv = triangular_solve(theta_2_chol, np.eye(M), lower=True)

    lambda_1 = cholesky_solve(theta_2_chol, theta_1)
    lambda_2 = -0.5*theta_2_chol_inv.T @ theta_2_chol_inv

    return lambda_1, lambda_2

@jit
def theta_to_lambda_diagonal(theta_1, theta_2):
    lambda_1 = theta_1 / theta_2
    lambda_2 = -0.5/theta_2

    return lambda_1, lambda_2

@jit
def lambda_to_theta_diagonal(lambda_1, lambda_2):
    theta_2 = 1/(-2*lambda_2)
    theta_1 = theta_2 * lambda_1

    return theta_1, theta_2

@jit
def lambda_to_theta(lambda_1, lambda_2):
    M = lambda_1.shape[0]
    jit = settings.ng_jitter * np.eye(M) 

    lambda_2_chol = cholesky(-2*lambda_2+jit)

    theta_2 =  cholesky_solve(lambda_2_chol, np.eye(M))
    theta_1 =  theta_2 @ lambda_1

    #theta_1 =  cholesky_solve(lambda_2_chol, lambda_1)

    return theta_1, theta_2

@jit
def xi_to_lambda(xi1, xi2):
    theta_1, theta_2 = xi_to_theta(xi1, xi2)
    return theta_to_lambda(theta_1, theta_2)

@jit
def lambda_to_xi(lambda_1, lambda_2):
    theta_1, theta_2 = lambda_to_theta(lambda_1, lambda_2)
    return theta_to_xi(theta_1, theta_2)

@jit
def xi_to_expectation(xi1, xi2):
    xi2 = xi2 @ xi2.T
    return xi1, xi1 @ xi1.T + xi2

@jit
def expectation_to_xi(mu1, mu2):
    M = mu1.shape[0]
    jit = settings.ng_jitter * np.eye(M) 

    xi2 = cholesky(mu2 - mu1 @ mu1.T + jit)

    return mu1, xi2



def natural_gradient_for_gaussian_approx_posterior(model, beta, approx_posterior, m_grad, s_grad):
    """
        Implments Natural gradients for q(u) with a general likelihood and Gaussian approximate posterior. 
            For further details see: 
                `Gaussian Processes for Big Data' - Hensman et al
                `Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models' - Salimbeni et al
        The approximate posterior is a Gaussian:
                q(u) = N(u | m, S), 
            where
                θ = (m, S)
            and is parameterised by
                
                ξ = (m, L) where S = LL^T
            with natural parameters:
                λ = (S⁻¹ m, - ½S⁻¹)
            and expectation parameters:
                
                μ = (m, mm^T + S⁻¹)
        The natural gradient step is given by, and taking the chain rule leads to:
            
            λᵣ₊₁ = λᵣ + β ∂L/∂μ
                 = λᵣ + β (∂L/∂ξ) (∂ξ/∂μ)
        This is a vector jacobian product because (∂L/∂ξ) is a vector.
        To update m, S we simply update λᵣ₊₁ and repameterise to get mᵣ₊₁, Sᵣ₊₁:
            mᵣ₊₁ = (- 2 λᵣ₊₁(2))⁻¹ λᵣ₊₁(1)
            Sᵣ₊₁ = (- 2 λᵣ₊₁(2))⁻¹
    """
    m = approx_posterior._m.value
    S_chol_flattened = approx_posterior._S_chol.value

    M = m.shape[0]

    S_chol = lower_triangle(S_chol_flattened, M)

    #Calculate natural and expectation parameters:
    theta_1, theta_2 = xi_to_theta(m, S_chol)
    lambda_1_init, lambda_2_init = theta_to_lambda(theta_1, theta_2)

    mu1, mu2 = xi_to_expectation(m, S_chol)

    partial_m = m_grad
    partial_s_chol_flattened = s_grad
    partial_s_chol = lower_triangle(partial_s_chol_flattened, M)

    #calculate ∂ξ/μ 
    x, u = vjp(expectation_to_xi, mu1, mu2)

    #calculate ∂L/μ = ∂L/∂ξ ∂ξ/μ
    u = u((partial_m, partial_s_chol))
    lambda_1, lambda_2 = u[0], u[1]

    #symmetrize gradient
    # This is the same problem as in gpytorch - see
    #   - https://github.com/pytorch/pytorch/issues/18825
    #   - and for the same solution `_cholesky_backward` here https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/variational/natural_variational_distribution.py 
    # TODO: double check that this is actually what is going on
    lambda_2 = lambda_2/2 
    lambda_2 = lambda_2 + lambda_2.T

    #gradient update
    #∂L/μ has been calculated with the negative ELBO however the natural gradients are defined on the orginal ELBO
    # hence we use the negative lambda's here
    lambda_1 = lambda_1_init + beta*(-lambda_1)

    lambda_2 = lambda_2_init + beta*(-lambda_2)

    #convert from natural parameters to the raw parameters
    theta_1, theta_2 = lambda_to_theta(lambda_1, lambda_2)

    xi1, xi2 = theta_to_xi(theta_1, theta_2)
    xi2 = xi2[np.tril_indices(M, 0)]

    return [xi1, xi2]



@dispatch('VGP', 'MeanFieldApproximatePosterior')
def natural_gradients(model, beta: float) -> np.ndarray:
    approx_posteriors = model.approximate_posterior.approx_posteriors
    num_q = len(approx_posteriors)

    param_dict = get_parameters(model, replace_name=False, return_id=True)

    m_name_list = []
    S_chol_list = []
    for q in approx_posteriors:

        m_name = get_var_name_with_id(model, id(q._m.raw_var), param_dict)
        S_chol_name = get_var_name_with_id(model, id(q._S_chol.raw_var), param_dict)

        m_name_list.append(m_name)
        S_chol_list.append(S_chol_name)

    # Precompute all gradients
    # Then vmap through them

    #calculate ∂L/∂ξ 
    vc = model.vars()

    # To use objax to compute gradients we have to pass a VarCollection
    # This requires knowning the (objax) id string
    approx_posterior_vars = [*m_name_list, *S_chol_list]

    # Extract only the approximate posterior variables to compute grads with
    vars_to_diff = vc_keep_vars(vc, approx_posterior_vars)

    # Construct the objax grad fucntions
    grad_fn = objax.GradValues(model.get_objective, vars_to_diff)
    gradients, _ = grad_fn()

    # Collect all m_grads and S_grads
    m_grads = []
    S_grads = []
    for q in range(0, len(gradients), 2):
        m_grads.append(gradients[q])
        S_grads.append(gradients[q+1])

    xi1_arr, xi2_arr = batch_or_loop(
        lambda m, b, q, m_grad, s_grad: natural_gradient_for_gaussian_approx_posterior(m, b, q, m_grad, s_grad),
        [model, beta, approx_posteriors, np.array(m_grads), np.array(S_grads)],
        [None, None, 0, 0, 0],
        dim = num_q,
        out_dim=2,
        batch_type = get_batch_type(approx_posteriors)
    )


    return xi1_arr, xi2_arr

@dispatch('VGP', 'FullGaussianApproximatePosterior')
def natural_gradients(model, beta: float) -> np.ndarray:
    q = model.approximate_posterior

    param_dict = get_parameters(model, replace_name=False, return_id=True)

    # Collect q parameters
    m_name = get_var_name_with_id(model, id(q._m.raw_var), param_dict)
    S_chol_name = get_var_name_with_id(model, id(q._S_chol.raw_var), param_dict)

    approx_posterior_vars = [m_name, S_chol_name]

    vc = model.vars()

    # Extract only the approximate posterior variables to compute grads with
    vars_to_diff = vc_keep_vars(vc, approx_posterior_vars)

    # Construct the objax grad fucntions
    grad_fn = objax.GradValues(model.get_objective, vars_to_diff)
    gradients, _ = grad_fn()
    m_grad = gradients[0]
    S_grad = gradients[1]

    xi1, xi2 = natural_gradient_for_gaussian_approx_posterior(model, beta, q, m_grad, S_grad)

    return xi1, xi2

@jit
def cvi_diagonal_update(Y_tilde, V_tilde, m, s, m_grad, s_grad, beta):
    # Get natural parameters for approximate likelihood
    lambda_1, lambda_2 = theta_to_lambda_diagonal(Y_tilde, V_tilde)

    # mu_grad and var_grad are ∂ell/∂θ 
    #calculate ∂ell/∂μ  = ∂ell/∂θ ∂θ/∂μ 
    grad_1 = m_grad - 2*s_grad*m
    grad_2 = s_grad

    # Natural gradient update updatae
    lambda_1_new  = (1-beta)*lambda_1 + beta* grad_1
    lambda_2_new  = (1-beta)*lambda_2 + beta* grad_2

    # Convert to theta
    theta_1, theta_2 = lambda_to_theta_diagonal(lambda_1_new, lambda_2_new)

    return theta_1[..., None], theta_2[..., None]

@jit
def cvi_block_update(Y_tilde, V_tilde, m, s, m_grad, s_grad, beta):
    # Get natural parameters for approximate likelihood
    lambda_1, lambda_2 = theta_to_lambda(Y_tilde, V_tilde)

    # mu_grad and var_grad are ∂ell/∂θ 
    #calculate ∂ell/∂μ  = ∂ell/∂θ ∂θ/∂μ 
    grad_1 = m_grad - 2*s_grad @ m
    grad_2 = s_grad

    # Natural gradient update updatae
    lambda_1_new  = (1-beta)*lambda_1 + beta* grad_1
    lambda_2_new  = (1-beta)*lambda_2 + beta* grad_2

    # Convert to theta
    theta_1, theta_2 = lambda_to_theta(lambda_1_new, lambda_2_new)

    return theta_1, theta_2

@jit
def reparametise_cholesky_grad(s, s_chol_grad, prior):
    #Calculate ∂L/μ = ∂L/∂ξ ∂ξ/μ 
    x, u = vjp(
        lambda A: cholesky(add_jitter(prior.unpermute_mat(A), settings.ng_jitter)), 
        s
    )
    s_grad = u(s_chol_grad)[0]

    # Symmetrize gradient (in case jax.scipy.linalg.cholesky is used)
    s_grad = s_grad/2 
    s_grad = s_grad + s_grad.T

    return s_grad

@jit
def reparametise_vec_grad(m, m_grad, prior):
    #Calculate ∂L/μ = ∂L/∂ξ ∂ξ/μ 
    x, u = vjp(
        lambda v: prior.unpermute_vec(v), 
        m
    )
    m_grad = u(m_grad)[0]

    return m_grad


@dispatch('VGP', ConjugateApproximatePosterior)
def natural_gradients(model, beta: float) -> np.ndarray:
    """
    Diagonal CVI Natural Gradients
    """
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

    Z_tiled = prior_Z(model.prior)

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

    mu_grads, var_chol_grads = jax.grad(partial_ell, (1, 2))(model, mu_arr, vectorized_lower_triangular_cholesky(var_arr))
    var_chol_grads = vectorized_lower_triangular(var_chol_grads, N=mu_grads[0].shape[0])

    # Make sure shapes are correct
    chex.assert_shape(Y_tilde_arr, mu_grads.shape)
    chex.assert_shape(Y_tilde_arr, mu_arr.shape)
    chex.assert_shape(V_tilde_arr, var_chol_grads.shape)
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
        var_chol_grads, 
        beta
    )

    chex.assert_shape(Y_tilde_arr, new_Y_tilde.shape)
    chex.assert_shape(V_tilde_arr, new_V_tilde.shape)

    new_V_tilde = new_V_tilde[:, None, ...]

    return new_Y_tilde, new_V_tilde


@dispatch('VGP', FullConjugateGaussian)
def natural_gradients(model, beta: float) -> np.ndarray:
    print('ng_jitter: ', settings.ng_jitter)
    # Get natural parameters
    q = model.approximate_posterior

    # Collect CVI parameters
    Y_tilde_arr, V_tilde_arr = q.surrogate.Y, q.surrogate.likelihood.variance

    Z_tiled = prior_Z(model.prior)
    Z = np.vstack(Z_tiled)

    M = Z_tiled[0].shape[0]
    Q = Z_tiled.shape[0]

    # These are in data-latent order
    Y_tilde_arr = np.reshape(Y_tilde_arr, [M*Q, 1])
    V_tilde_arr = V_tilde_arr[0]

    # Predict in data-latent order
    q_mu_z, q_var_z = q.surrogate.predict_blocks(Z_tiled, M, M*Q, diagonal=False)
    q_mu_z, q_var_z = q_mu_z[0][..., None], q_var_z[0]

    prior = q.surrogate.prior

    # FullGaussianApproximatePosterior is meant to be used in latent-data order
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
    s_grad_permuted = reparametise_cholesky_grad(q_var_z, var_chol_grads, prior)
    mu_grads_permuted = reparametise_vec_grad(q_mu_z, mu_grads, prior)

    new_Y_tilde, new_V_tilde = cvi_block_update(
        Y_tilde_arr, V_tilde_arr, q_mu_z, q_var_z, mu_grads_permuted, s_grad_permuted, beta
    )
    # Fix dimensions and order
    new_Y_tilde = np.reshape(new_Y_tilde, q.surrogate.Y.shape)
    new_V_tilde = new_V_tilde[None, ...]


    return new_Y_tilde, new_V_tilde






