from ..settings import jitter
from ..kernels import Kernel, RBF
from ..likelihood import Gaussian
from ..approximate_posteriors import GaussianApproximatePosterior
from ..dispatch import dispatch
from .gaussian import log_gaussian
from ..batching import Batched
from .. import utils
from .matrix_ops import cholesky, log_chol_matrix_det, add_jitter, diagonal_from_cholesky

import jax
from jax import jit
import jax.numpy as np
import chex
from typing import List
from objax import ModuleList

@jit
def gaussian_kl( mu_1, mu_2, covar_chol_1, covar_chol_2) -> np.ndarray:
    """
    computs KL[g1, g2]
    """

    log_det_term = log_chol_matrix_det(covar_chol_2) - log_chol_matrix_det(covar_chol_1)

    # see https://github.com/GPflow/GPflow/blob/develop/gpflow/kullback_leiblers.py
    trace_term = np.sum(
        np.square(
            jax.scipy.linalg.solve_triangular(covar_chol_2, covar_chol_1, lower=True)
        )
    )

    err = mu_2 - mu_1

    maha_term = np.sum(
        np.square(jax.scipy.linalg.solve_triangular(covar_chol_2, err, lower=True))
    )

    N = mu_1.shape[0] * 1.0

    return 0.5 * (log_det_term - N + trace_term + maha_term)


@jit
def whitened_gaussian_kl(mu_1, covar_chol_1) -> np.ndarray:
    """
    Assumes that g2 is a standard Gaussian - N(0, I)
    """

    covar_1_diag = diagonal_from_cholesky(covar_chol_1)

    log_det_term = -log_chol_matrix_det(covar_chol_1)

    trace_term = np.sum(covar_1_diag)

    #trace_term = np.trace(np.square(covar_chol_1))

    maha_term = np.sum(np.square(mu_1))

    N = mu_1.shape[0] * 1.0

    return 0.5 * (log_det_term - N + trace_term + maha_term)

@dispatch(object, GaussianApproximatePosterior, object, object)
def KL(X, approximate_posterior, kernel, sparsity):
    mu_1 = approximate_posterior.m
    covar_chol_1 = approximate_posterior.S_chol

    mu_2 = np.zeros(mu_1.shape)

    covar_2 = kernel.K(X, X)
    covar_chol_2 = cholesky(add_jitter(covar_2, jitter))

    return gaussian_kl(mu_1, mu_2, covar_chol_1, covar_chol_2)

@dispatch(object, GaussianApproximatePosterior, object, object)
def whitened_KL(X, approximate_posterior, kernel, sparsity):

    mu_1 = approximate_posterior.m
    covar_chol_1 = approximate_posterior.S_chol

    return whitened_gaussian_kl(mu_1, covar_chol_1)

