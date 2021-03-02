"""
    Consider a Gaussian:

        p(f) = N(f | m, S), 

    with standard parameters

        θ = (m, S)

    and natural parameters:

        λ = (S⁻¹ m, - ½S⁻¹)

    and expectation parameters:
        
        μ = (m, mm^T + S⁻¹)

    This file contains helper functions to map between these parameterisations
"""


from .. import Settings

from .general import lower_triangle, triangular_solve, cholesky_solve, cholesky

import jax
import jax.numpy as np
from jax.config import config
from jax import jit, partial
from jax import jacfwd, jacrev, vjp, jvp

import numpy as onp

import typing
from typing import List, Callable, Optional, Tuple


def standard_to_natural(mean, var):
    M = mean.shape[0]
    jit = Settings.jitter * np.eye(M)

    var_chol = cholesky(var + jit)
    var_chol_inverse = triangular_solve(var_chol, np.eye(M), lower=True)

    # S⁻¹ m
    lambda_1 = cholesky_solve(var_chol, mean)

    # - ½S⁻¹
    lambda_2 = -0.5 * var_chol_inverse.T @ var_chol_inverse

    return lambda_1, lambda_2


def standard_to_expectation(mean, var):
    # m, mm^T + S⁻¹
    return mean, mean @ mean.T + var


def natural_to_standard(lambda_1, lambda_2):
    """
    Given λ(1), λ(2) then
        S = - ½λ(2)⁻¹
        m = - ½λ(2)⁻¹λ(1)
    """
    M = lambda_1.shape[0]
    jit = Settings.jitter * np.eye(M)

    lambda_2_chol = cholesky(-2 * lambda_2 + jit)

    lambda_2_chol_inv = triangular_solve(lambda_2_chol, np.eye(M), lower=True)

    theta_1 = cholesky_solve(lambda_2_chol, lambda_1)
    theta_2 = lambda_2_chol_inv.T @ lambda_2_chol_inv

    return theta_1, theta_2


def natural_parameterised_by_covariance_to_mean(lambda_1, covariance):
    return covariance @ lambda_1


def natural_to_standard_diagonal(lambda_1, lambda_2):
    # fast implemenationi for when lambda_2 is vector. return a vector variance
    M = lambda_1.shape[0]

    lambda_2_inv = 1 / (-2 * lambda_2)

    theta_1 = np.multiply(lambda_2_inv, lambda_1)
    theta_2 = lambda_2_inv

    return theta_1, theta_2


def expectation_to_standard(mu_1, mu_2):
    raise NotImplementedError()
