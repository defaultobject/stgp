from ..module import Module
from ..dispatcher import Dispatcher
from ..data import Data, ListData
from ..sparsity import Sparsity
from ..distributions import *
from ..likelihoods import Likelihood
from ..models import *
from ..inference import *
from ..decorators import *


from .. import Settings

from .. import Parameter

from .general import (
    lower_triangle,
    triangular_solve,
    cholesky_solve,
    cholesky,
    positive_transform,
    inv_positive_transform,
    flatten_cholesky,
)

import jax
import jax.numpy as np
from jax.config import config
from jax import jit, partial
from jax import jacfwd, jacrev, vjp, jvp

import numpy as onp

import typing
from typing import List, Callable, Optional, Tuple

"""
    Inside these functions there is an optional jitter Setting.nat_grad_jitter
    Sometimes the natural gradients are more stable without jitter.
"""


def xi_to_theta(xi1, xi2):
    # theta is parameterised by the cholesky decomposition
    # xi2 = lower_triangle(xi2, xi1.shape[0])
    return xi1, xi2 @ xi2.T


def theta_to_xi(theta_1, theta_2):
    M = theta_1.shape[0]
    jit = Settings.jitter * np.eye(M) * Settings.nat_grad_jitter

    xi1 = theta_1
    xi2 = cholesky(theta_2 + jit)

    return xi1, xi2


def theta_to_lambda(theta_1, theta_2):
    M = theta_1.shape[0]
    jit = Settings.jitter * np.eye(M) * Settings.nat_grad_jitter

    theta_2_chol = cholesky(theta_2 + jit)

    theta_2_chol_inv = triangular_solve(theta_2_chol, np.eye(M), lower=True)

    lambda_1 = cholesky_solve(theta_2_chol, theta_1)
    lambda_2 = -0.5 * theta_2_chol_inv.T @ theta_2_chol_inv

    return lambda_1, lambda_2


def lambda_to_theta(lambda_1, lambda_2):
    M = lambda_1.shape[0]
    jit = Settings.jitter * np.eye(M) * Settings.nat_grad_jitter

    lambda_2_chol = cholesky(-2 * lambda_2 + jit)

    lambda_2_chol_inv = triangular_solve(lambda_2_chol, np.eye(M), lower=True)

    theta_1 = cholesky_solve(lambda_2_chol, lambda_1)
    theta_2 = lambda_2_chol_inv.T @ lambda_2_chol_inv

    return theta_1, theta_2


def xi_to_lambda(xi1, xi2):
    theta_1, theta_2 = xi_to_theta(xi1, xi2)
    return theta_to_lambda(theta_1, theta_2)


def lambda_to_xi(lambda_1, lambda_2):
    theta_1, theta_2 = lambda_to_theta(lambda_1, lambda_2)
    return theta_to_xi(theta_1, theta_2)


def xi_to_expectation(xi1, xi2):
    # xi2 = lower_triangle(xi2, xi1.shape[0])
    xi2 = xi2 @ xi2.T
    return xi1, xi1 @ xi1.T + xi2


def expectation_to_xi(mu1, mu2):
    M = mu1.shape[0]
    jit = Settings.jitter * np.eye(M) * Settings.nat_grad_jitter

    xi2 = cholesky(mu2 - mu1 @ mu1.T + jit)
    # xi2 = xi2[np.tril_indices(xi2.shape[0], 0)]

    return mu1, xi2


def match_contains(arr, s):
    return [a for a in arr if s in a]


# @Dispatcher.register('natural_gradients', None)
@jit
def _gaussian_ell_natural_gradients(model: Model, params) -> np.ndarray:

    param_names = params.keys()

    m_param_name = match_contains(param_names, "mu")[0]
    s_chol_param_name = match_contains(param_names, "covariance")[0]

    m = params[m_param_name]
    # m = np.squeeze(m)

    M = m.shape[0]

    S_chol_flattened = params[s_chol_param_name]
    S_chol = lower_triangle(S_chol_flattened, M)

    theta_1, theta_2 = xi_to_theta(m, S_chol)
    lambda_1_init, lambda_2_init = theta_to_lambda(theta_1, theta_2)

    mu1, mu2 = xi_to_expectation(m, S_chol)

    M = theta_1.shape[0]
    jit = Settings.jitter * np.eye(M)

    base_model = model.model
    print(base_model)
    kernel = base_model.kernel

    X = base_model.data.X[0]
    Y = base_model.data.Y[0]
    # TODO
    Z = base_model.data.X[0]

    K_xx = kernel.K(X, X)
    K_xz = kernel.K(X, Z)
    K_zz = kernel.K(Z, Z)
    K_zz_chol = cholesky(K_zz + jit)

    lik_noise = base_model.likelihood.variance
    inv_lik_noise = 1 / base_model.likelihood.variance

    beta = 1.0

    K_zz_chol_inv = triangular_solve(K_zz_chol, np.eye(K_zz_chol.shape[0]), lower=True)
    prec = inv_lik_noise * np.eye(X.shape[0]) + K_zz_chol_inv.T @ K_zz_chol_inv
    lambda_2_lik = -0.5 * prec + lambda_2_init
    lambda_2 = lambda_2_init + beta * lambda_2_lik
    print(lambda_2_lik)
    exit()

    # lambda_2 = -0.5*inv_lik_noise*np.eye(X.shape[0])-0.5*K_xx

    # lambda_1_lik = inv_lik_noise*np.eye(X.shape[0]) @ cholesky_solve(K_zz_chol, K_xz.T) @ Y - cholesky_solve(S_chol, m)
    lambda_1_lik = inv_lik_noise * np.eye(X.shape[0]) @ Y - lambda_1_init
    lambda_1 = lambda_1_init + beta * lambda_1_lik

    # lambda_1 = inv_lik_noise*np.eye(X.shape[0]) @  Y
    # lambda_2 = -0.5*prec

    # lambda_1 = lambda_1_init + beta*(inv_lik_noise*np.eye(X.shape[0]) @ Y - cholesky_solve(S_chol, m))

    theta_1, theta_2 = lambda_to_theta(lambda_1, lambda_2)

    xi1, xi2 = theta_to_xi(theta_1, theta_2)
    xi2 = xi2[np.tril_indices(M, 0)]

    if False:
        # GP prediction
        chol = cholesky(K_zz + lik_noise * np.eye(K_zz.shape[0]))
        xi1 = K_xx @ cholesky_solve(chol, Y)

        xi2 = (
            lik_noise
            * np.eye(K_zz.shape[0])
            @ K_xx
            @ cholesky_solve(chol, np.eye(K_zz.shape[0]))
        )
        xi2_chol = cholesky(xi2 + jit)
        xi2 = xi2_chol[np.tril_indices(M, 0)]
        print(xi2)

    return {m_param_name: xi1, s_chol_param_name: xi2}


def psd_retraction_map(sigma, b):
    chol = cholesky(sigma + Settings.jitter * np.eye(sigma.shape[0]))
    sigma_new = sigma + b + 0.5 * b @ cholesky_solve(chol, b)

    return sigma_new


@Dispatcher.register("natural_gradients", GP, None, None)
@jit_with_scope(static_argnums=[1])
def general_ell_natural_gradients(model: Model, beta: float) -> np.ndarray:
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

    if beta is None:
        beta = 0.1
        warnings.warn(
            "beta not specified. Using default number of {beta}.".format(beta=beta)
        )

    params = Parameter.PARAM_DICT

    # Collect variational parameters
    m_param_name = model.base_model.inference.variational_posterior.components[
        0
    ].distribution.mu.name
    s_chol_param_name = model.base_model.inference.variational_posterior.components[
        0
    ].distribution.covariance_chol.name

    m = model.base_model.inference.variational_posterior.components[
        0
    ].distribution.mu.raw
    S_chol_flattened = model.base_model.inference.variational_posterior.components[
        0
    ].distribution.covariance_chol.raw

    M = m.shape[0]

    S_chol = lower_triangle(S_chol_flattened, M)

    # Calculate natural and expectation parameters:
    theta_1, theta_2 = xi_to_theta(m, S_chol)
    lambda_1_init, lambda_2_init = theta_to_lambda(theta_1, theta_2)

    mu1, mu2 = xi_to_expectation(m, S_chol)

    # calculate ∂L/∂ξ - this is already computed by the model
    # TODO: only need gradients wrt the approximate posterior, not all params
    _, gradients = model.get_objective(params=params)

    partial_m = gradients[m_param_name]
    partial_s_chol_flattened = gradients[s_chol_param_name]
    partial_s_chol = lower_triangle(partial_s_chol_flattened, M)

    # calculate ∂ξ/μ
    x, u = vjp(expectation_to_xi, mu1, mu2)

    # calculate ∂L/μ = ∂L/∂ξ ∂ξ/μ
    u = u((partial_m, partial_s_chol))
    lambda_1, lambda_2 = u[0], u[1]

    # symmetrize gradient
    # This is the same problem as in gpytorch - see
    #   - https://github.com/pytorch/pytorch/issues/18825
    #   - and for the same solution `_cholesky_backward` here https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/variational/natural_variational_distribution.py
    # TODO: double check that this is actually what is going on
    lambda_2 = lambda_2 / 2
    lambda_2 = lambda_2 + lambda_2.T

    # gradient update
    # ∂L/μ has been calculated with the negative ELBO however the natural gradients are defined on the orginal ELBO
    # hence we use the negative lambda's here
    lambda_1 = lambda_1_init + beta * (-lambda_1)

    if Settings.enforce_psd:
        # -2*lambda_2_init = S^{-1} and hence is psd
        lambda_2 = psd_retraction_map(-2 * lambda_2_init, -2 * beta * (-lambda_2)) / (
            -2
        )
    else:
        lambda_2 = lambda_2_init + beta * (-lambda_2)

    # convert from natural parameters to the raw parameters
    theta_1, theta_2 = lambda_to_theta(lambda_1, lambda_2)

    if False:
        print("lambda_1: ", lambda_1)
        print("lambda_2: ", lambda_2)
        print("theta_1: ", theta_1)
        print("theta_2: ", theta_2)

    xi1, xi2 = theta_to_xi(theta_1, theta_2)
    xi2 = xi2[np.tril_indices(M, 0)]

    return {m_param_name: xi1, s_chol_param_name: xi2}


@Dispatcher.register("natural_gradients", GP, StateSpaceVI, 1)
@jit_with_scope(static_argnums=[1])
def cvi_diagonal_natural_gradients(
    model: Model, beta: Optional[float] = None
) -> np.ndarray:
    """
    The approximate posterior is parameterised by:
        lambda_1 in Nx1
        variance in Nx1
    """
    if beta is None:
        beta = 0.1
        warnings.warn(
            "beta not specified. Using default number of {beta}.".format(beta=beta)
        )

    # Collect variational parameters
    lambda_1_name = (
        model.base_model.inference.variational_posterior.distribution.lambda_1.name
    )
    variance_name = (
        model.base_model.inference.variational_posterior.distribution.covariance.name
    )

    lambda_1 = (
        model.base_model.inference.variational_posterior.distribution.lambda_1.raw
    )
    variance = (
        model.base_model.inference.variational_posterior.distribution.covariance.raw
    )

    variance = np.clip(variance, 1e-10)

    lambda_2 = -0.5 * (1 / variance)

    m, s = model.base_model.get_m_s(predict=False)

    # calculate ∂ell/∂θ
    m_grad, s_grad = jax.grad(model.get_ell_term_wrt_m_s, (0, 1))(m, s)

    # calculate ∂ell/∂μ  = ∂ell/∂θ ∂θ/∂μ
    grad_1 = m_grad - 2 * s_grad * m
    grad_2 = s_grad

    lambda_1_new = (1 - beta) * lambda_1 + beta * grad_1
    lambda_2_new = (1 - beta) * lambda_2 + beta * grad_2

    variance_new = inv_positive_transform(-0.5 * (1 / lambda_2_new))

    return {lambda_1_name: lambda_1_new, variance_name: variance_new}


@Dispatcher.register("natural_gradients", GP, StateSpaceVI, None)
@jit_with_scope(static_argnums=[1])
def cvi_block_diagonal_natural_gradients(model: Model, beta: float) -> np.ndarray:
    """
    The approximate posterior is parameterised by:
        lambda_1 in Nx1
        variance in Nx1
    """
    if beta is None:
        beta = 0.1
        warnings.warn(
            "beta not specified. Using default number of {beta}.".format(beta=beta)
        )

    lambda_1_name = (
        model.base_model.inference.variational_posterior.distribution.lambda_1.name
    )

    covar_chol_name = (
        model.base_model.inference.variational_posterior.distribution.covar_chol.name
    )

    params = Parameter.PARAM_DICT

    lambda_1 = params[lambda_1_name]
    covar_chol = params[covar_chol_name]

    # return {lambda_1_name: lambda_1, covar_chol_name:covar_chol}

    covar_chol = jax.vmap(lower_triangle, in_axes=(0, None), out_axes=0)(
        covar_chol, lambda_1.shape[1]
    )

    variance_inv = jax.vmap(block_wise_cholesky_inverse, in_axes=(0), out_axes=0)(
        covar_chol
    )

    lambda_2 = -0.5 * variance_inv

    m, s = model.base_model.get_m_s(predict=False)

    # calculate ∂ell/∂θ
    m_grad, s_grad = jax.grad(model.get_ell_term_wrt_m_s, (0, 1))(m, s)

    # calculate ∂ell/∂μ  = ∂ell/∂θ ∂θ/∂μ
    res = jax.vmap(block_wise_matrix_mult, in_axes=(0, 0), out_axes=0)(s_grad, m)
    res = res[..., 0]

    grad_1 = m_grad - 2 * res
    grad_2 = s_grad

    grad_1 = np.expand_dims(grad_1, -1)

    lambda_1_new = (1 - beta) * lambda_1 + beta * grad_1

    if Settings.enforce_psd:
        block_retraction = jax.vmap(psd_retraction_map, in_axes=(0, 0), out_axes=(0))
        lambda_2_new = block_retraction(
            -2 * (1 - beta) * lambda_2, -2 * beta * grad_2
        ) / (-2)
    else:
        lambda_2_new = (1 - beta) * lambda_2 + beta * grad_2

    lambda_2_new_chol = jax.vmap(block_wise_cholesky, in_axes=(0), out_axes=0)(
        -2 * lambda_2_new
    )
    variance_new = jax.vmap(block_wise_cholesky_inverse, in_axes=(0), out_axes=0)(
        lambda_2_new_chol
    )

    variance_new_flattened = jax.vmap(
        block_wise_cholesky_flattened, in_axes=(0), out_axes=0
    )(variance_new)

    return {lambda_1_name: lambda_1_new, covar_chol_name: variance_new_flattened}


@jit
def block_wise_cholesky_multiply(var_chol):
    return var_chol @ var_chol.T


@jit
def block_wise_cholesky_flattened(var_chol):
    return flatten_cholesky(cholesky(var_chol), var_chol.shape[0])


@jit
def block_wise_cholesky_inverse(var_chol):
    return cholesky_solve(var_chol, np.eye(var_chol.shape[0]))


@jit
def block_wise_cholesky(var):
    return cholesky(var + Settings.jitter * np.eye(var.shape[0]))


@jit
def block_wise_matrix_mult(block_mat_1, block_mat_2):
    res = block_mat_1 @ block_mat_2[:, None]

    return res
