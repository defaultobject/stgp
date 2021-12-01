"""Standard Gaussian methods."""

import jax
import jax.numpy as np
from jax import jit
import chex

from .matrix_ops import cholesky, cholesky_solve, log_chol_matrix_det


@jit
def log_gaussian(Y, mu, sigma):
    # ensure matrices
    chex.assert_rank(mu, 2)
    chex.assert_rank(Y, 2)
    chex.assert_rank(sigma, 2)

    # ensure square matrix
    chex.assert_equal(sigma.shape[0], sigma.shape[1])

    jitter = 1e-5

    sigma_chol = cholesky(sigma + jitter * np.eye(sigma.shape[0]))

    N = Y.shape[0]

    c1 = -0.5 * N * np.log(2 * np.pi) 
    c2 = - 0.5 * log_chol_matrix_det(sigma_chol)
    c = c1+c2

    err = Y - mu
    mahal = err.T @ cholesky_solve(sigma_chol, err)

    ml = c - 0.5 * mahal
    return np.squeeze(ml)


@jit
def log_gaussian_scalar(Y, mu, variance):
    # ensure scalar
    chex.assert_rank(mu, 0)
    chex.assert_rank(Y, 0)
    chex.assert_rank(variance, 0)

    # scalar
    N = 1

    c1 = -0.5 * N * np.log(2 * np.pi) - N * 0.5 * np.log(variance)

    err = Y - mu
    mahal = (err * err) / variance

    ll = c1 - 0.5 * mahal

    chex.assert_rank(ll, 0)

    return ll


@jit
def log_gaussian_diagonal(Y, mu, variance):
    # TODO: just vmap log_gaussian_scalar

    raise NotImplementedError()

    N = Y.shape[0]

    # log |diag(var)| = \sum \log variance
    log_det = np.sum(np.log(variance))

    c1 = -0.5 * N * np.log(2 * np.pi) - 0.5 * log_det

    err = Y - mu
    inv_variance = 1 / variance
    mahal = err.T @ np.multiply(inv_variance, err)

    return c1 - 0.5 * mahal
