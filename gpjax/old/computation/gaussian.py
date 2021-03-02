from jax.config import config

config.update("jax_enable_x64", True)
import jax
import jax.numpy as np

from jax import jit, partial

from ..settings import Settings
from .general import cholesky_solve, log_chol_matrix_det, cholesky


def log_gaussian(Y, mu, sigma):
    sigma_chol = cholesky(sigma + Settings.jitter * np.eye(sigma.shape[0]))

    N = Y.shape[0]

    c1 = -0.5 * N * np.log(2 * np.pi) - 0.5 * log_chol_matrix_det(sigma_chol)

    err = Y - mu
    mahal = err.T @ cholesky_solve(sigma_chol, err)

    ml = c1 - 0.5 * mahal
    return np.squeeze(ml)


def log_gaussian_diagonal(Y, mu, variance):

    N = Y.shape[0]

    # log |diag(var)| = \sum \log variance
    log_det = np.sum(np.log(variance))

    c1 = -0.5 * N * np.log(2 * np.pi) - 0.5 * log_det

    err = Y - mu
    inv_variance = 1 / variance
    mahal = err.T @ np.multiply(inv_variance, err)

    return c1 - 0.5 * mahal


def log_gaussian_scalar(Y, mu, variance):

    N = Y.shape[0]

    c1 = -0.5 * N * np.log(2 * np.pi) - N * 0.5 * np.log(variance)

    err = Y - mu
    mahal = (err.T @ err) / variance

    return c1 - 0.5 * mahal


def gaussian_conditional(k_xx, k_xz, k_zz, mu, sig):
    k_zz_chol = np.linalg.cholesky(k_zz + Settings.jitter * np.eye(k_zz.shape[0]))

    A = cholesky_solve(k_zz_chol, k_xz.T)
    mu = k_xz @ cholesky_solve(k_zz_chol, mu)
    sig = k_xx - k_xz @ A + A.T @ sig @ A

    return mu, sig


def diagional_gaussian_conditional(k_xx_diag, k_xz, k_zz, mu, sig):
    """
    Returns only the diagional of the full covariance
    """
    k_zz_chol = np.linalg.cholesky(k_zz + Settings.jitter * np.eye(mu.shape[0]))
    k_sig_chol = np.linalg.cholesky(sig + Settings.jitter * np.eye(mu.shape[0]))

    mu = k_xz @ cholesky_solve(k_zz_chol, mu)

    A1 = cholesky_solve(k_zz_chol, k_xz.T)
    A2 = k_xz @ cholesky_solve(k_zz_chol, k_sig_chol)

    sig = k_xx_diag - np.sum(np.square(A1), axis=0) + np.sum(np.square(A2), axis=1)

    return mu, sig


def whitened_gaussian_conditional(k_xx, k_xz, k_zz, mu, sig):
    k_zz_chol = np.linalg.cholesky(k_zz + Settings.jitter * np.eye(mu.shape[0]))

    mu = k_xz @ jax.scipy.linalg.solve_triangular(k_zz_chol.T, mu, lower=False)

    sig = k_xx - k_xz @ cholesky_solve(k_zz_chol, k_xz.T)
    A = k_xz @ jax.scipy.linalg.solve_triangular(k_zz_chol.T, sig, lower=False)
    sig = sig + A @ A.T

    return mu, sig


def diagional_whitened_gaussian_conditional(k_xx_diag, k_xz, k_zz, mu, sig):
    """
    Returns only the diagional of the full covariance
    """
    k_zz_chol = np.linalg.cholesky(k_zz + Settings.jitter * np.eye(mu.shape[0]))
    sig_chol = np.linalg.cholesky(sig + Settings.jitter * np.eye(mu.shape[0]))

    mu = k_xz @ jax.scipy.linalg.solve_triangular(k_zz_chol.T, mu, lower=False)

    A1 = jax.scipy.linalg.solve_triangular(k_zz_chol, k_xz.T, lower=True)
    A2 = k_xz @ jax.scipy.linalg.solve_triangular(k_zz_chol.T, sig_chol, lower=False)

    sig = k_xx_diag - np.sum(np.square(A1), axis=0) + np.sum(np.square(A2), axis=1)

    return mu, sig
