"""Standard Gaussian methods."""

import jax
import jax.numpy as np
from jax import jit
import chex
from .. import settings

from .matrix_ops import cholesky, cholesky_solve, log_chol_matrix_det
from ..utils.nan_utils import mask_to_identity, get_mask, mask_vector

@jit
def log_gaussian(Y, mu, sigma):
    """
    Computes 
        log N(Y | m, S) = -(1/2) [N log 2π + log |S| + (Y-m)^T S⁻¹ (Y-m)]
    """
    # ensure matrices
    chex.assert_rank(mu, 2)
    chex.assert_rank(Y, 2)
    chex.assert_rank(sigma, 2)

    # ensure square matrix
    chex.assert_equal(sigma.shape[0], sigma.shape[1])

    sigma_chol = cholesky(sigma + settings.jitter * np.eye(sigma.shape[0]))

    N = Y.shape[0]

    c1 = -0.5 * N * np.log(2 * np.pi) 
    c2 = - 0.5 * log_chol_matrix_det(sigma_chol)
    c = c1+c2

    err = Y - mu
    mahal = err.T @ cholesky_solve(sigma_chol, err)

    ml = c - 0.5 * mahal
    return np.squeeze(ml) 


@jit
def log_gaussian_with_mask(Y, mu, sigma, mask):
    """
    Let Y_m, Y_o indicate the missing and observed datapoints then this function computes 
        log N(Y | m, S) = log N(Y_o | m_o, S_o) N(Y_m | m_m, S_m)

    Then the log-marginal likelihood only on Y_o is given by:
        log N(Y | m, S) - log N(Y_m | 0, 1) where Y_m = 0.
    which is equal to
        log N(Y | m, S) + (1/2) N_m log 2π

    We work with both Y_m and Y_o to keep matrix dimensions fixed no matter how many missing observations
        there are.
    """
    # ensure matrices
    chex.assert_rank(mu, 2)
    chex.assert_rank(Y, 2)
    chex.assert_rank(sigma, 2)
    chex.assert_rank(mask, 1)

    # ensure square matrix
    chex.assert_equal(sigma.shape[0], sigma.shape[1])

    # masking
    Y = np.nan_to_num(Y, nan=0.0)
    sigma = mask_to_identity(sigma, mask)
    mu = mask_vector(mu, mask)

    sigma_chol = cholesky(sigma + settings.jitter * np.eye(sigma.shape[0]))

    N = Y.shape[0]

    c1 = -0.5 * N * np.log(2 * np.pi) 
    c2 = - 0.5 * log_chol_matrix_det(sigma_chol)
    c = c1+c2

    err = Y - mu
    mahal = err.T @ cholesky_solve(sigma_chol, err)

    ml = c - 0.5 * mahal
    # MASK is one if non nan, zero is nan

    N_mask = np.sum(1-mask)

    log_n = np.log(np.clip(N_mask, 1.0, None))

    return np.sum(np.squeeze(ml)) + 0.5 * (N_mask * np.log(2* np.pi))

@jit
def log_gaussian_with_nans(Y, mu, sigma):
    mask = get_mask(Y)
    return log_gaussian_with_mask(Y, mu, sigma, mask)

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
    return jax.vmap(
        log_gaussian_scalar, 
        [0, 0, 0]
    )(np.squeeze(Y), np.squeeze(mu), np.squeeze(variance))

