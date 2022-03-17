from ... import settings
from ..matrix_ops import cholesky, cholesky_solve, triangular_solve, add_jitter, lower_triangle, vectorized_lower_triangular_cholesky, vectorized_lower_triangular, lower_triangular_cholesky, lower_triangle
from ...utils.utils import vc_keep_vars, get_parameters, get_var_name_with_id, get_batch_type
from ...dispatch import dispatch, evoke
from ...approximate_posteriors import ConjugateApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullConjugateGaussian, FullGaussianApproximatePosterior

from ..elbos.elbos import compute_expected_log_liklihood
from .exponential_family_transforms import xi_to_theta, theta_to_lambda, xi_to_expectation, expectation_to_xi, lambda_to_theta, theta_to_xi

import chex
from batchjax import batch_or_loop, BatchType
import jax
import jax.numpy as np
from jax import grad, jit, jacfwd
from jax import vjp

import objax

from typing import List

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






