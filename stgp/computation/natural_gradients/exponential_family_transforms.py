from ... import settings
from ..matrix_ops import cholesky, cholesky_solve, triangular_solve, add_jitter, lower_triangle, vectorized_lower_triangular_cholesky, vectorized_lower_triangular, lower_triangular_cholesky, lower_triangle

import jax
import jax.numpy as np
from jax import  jit
import chex

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



