from .. import Kernel
from .. import Likelihood

from .general import log_chol_matrix_det, cholesky_solve

import typing

import jax
import jax.numpy as np

from jax import jit, partial


@partial(jit, static_argnums=(0, 1))
def log_marginal_likelihood(
    X: np.ndarray, Y: np.ndarray, kernel: Kernel, likelihood: Likelihood
) -> np.ndarray:
    N = X.shape[0]
    k_xx = kernel.K(X, X)
    lik_noise = likelihood.variance
    k = k_xx + lik_noise * np.eye(N)
    k_chol = jax.scipy.linalg.cholesky(k, lower=True)

    ml = (
        -0.5 * Y.T @ cholesky_solve(k_chol, Y)
        - 0.5 * log_chol_matrix_det(k)
        - 0.5 * N * np.log(2 * np.pi)
    )
    return np.squeeze(ml)  # dim 1


@partial(jit, static_argnums=(0, 1))
def predict_y(
    XS: np.ndarray, X: np.ndarray, Y: np.ndarray, kernel: Kernel, likelihood: Likelihood
):
    N = X.shape[0]

    K_xs = kernel.K(XS, XS)
    K_xs_x = kernel.K(XS, X)
    K_x_x = kernel.K(X, X)

    k_xx = kernel.K(X, X)
    lik_noise = likelihood.variance
    k = k_xx + lik_noise * np.eye(N)
    k_chol = jax.scipy.linalg.cholesky(k, lower=True)

    mu = K_xs_x @ cholesky_solve(k_chol, Y)
    sig = K_xs - K_xs_x @ cholesky_solve(k_chol, K_xs_x.T)

    return mu, sig
