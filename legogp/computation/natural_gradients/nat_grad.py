from ...settings import jitter
from ..matrix_ops import cholesky, cholesky_solve, triangular_solve, add_jitter, lower_triangle
from ...utils.utils import vc_keep_vars


import jax
import jax.numpy as np
from jax import grad, jit
from jax import vjp

import objax

from typing import List


@jit
def xi_to_theta(xi1, xi2):
    return xi1, xi2 @ xi2.T

@jit
def theta_to_xi(theta_1, theta_2):
    M = theta_1.shape[0]
    jit = jitter * np.eye(M) 

    xi1 = theta_1
    xi2 = cholesky(theta_2+jit)

    return xi1, xi2

@jit
def theta_to_lambda(theta_1, theta_2):
    M = theta_1.shape[0]
    jit = jitter * np.eye(M) 

    theta_2_chol = cholesky(theta_2+jit)

    theta_2_chol_inv = triangular_solve(theta_2_chol, np.eye(M), lower=True)

    lambda_1 = cholesky_solve(theta_2_chol, theta_1)
    lambda_2 = -0.5*theta_2_chol_inv.T @ theta_2_chol_inv

    return lambda_1, lambda_2

@jit
def lambda_to_theta(lambda_1, lambda_2):
    M = lambda_1.shape[0]
    jit = jitter * np.eye(M) 

    lambda_2_chol = cholesky(-2*lambda_2+jit)

    lambda_2_chol_inv = triangular_solve(lambda_2_chol, np.eye(M), lower=True)

    theta_1 =  cholesky_solve(lambda_2_chol, lambda_1)
    theta_2 =  lambda_2_chol_inv.T @ lambda_2_chol_inv

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
    jit = jitter * np.eye(M) 

    xi2 = cholesky(mu2 - mu1 @ mu1.T + jit)

    return mu1, xi2




def general_ell_natural_gradients(model, beta: float, approx_posterior_vars: List[str]) -> np.ndarray:
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

    # Collect variational parameters

    # assume that we are in the single latent case
    # TODO: generalize
    approx_posterior = model.approximate_posterior.approx_posteriors[0]

    m = approx_posterior._m.value
    S_chol_flattened = approx_posterior._S_chol.value

    M = m.shape[0]

    S_chol = lower_triangle(S_chol_flattened, M)

    #Calculate natural and expectation parameters:
    theta_1, theta_2 = xi_to_theta(m, S_chol)
    lambda_1_init, lambda_2_init = theta_to_lambda(theta_1, theta_2)

    mu1, mu2 = xi_to_expectation(m, S_chol)

    #calculate ∂L/∂ξ - this is already computed by the model
    vc = model.vars()

    #TODO: this should not be hardcoded
    m_name = approx_posterior_vars[0]
    s_chol_name = approx_posterior_vars[1]

    vars_to_diff = vc_keep_vars(vc, approx_posterior_vars)

    grad_fn = objax.GradValues(model.get_objective, vars_to_diff)

    gradients, _ = grad_fn()

    partial_m = gradients[0]
    partial_s_chol_flattened = gradients[1]
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

