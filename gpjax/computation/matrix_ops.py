"""Standard operations on matrices."""

import jax.numpy as np
from jax import jit
import chex


@jit
def log_chol_matrix_det(chol):
    # ensure square matrix
    chex.assert_rank(chol, 2)
    chex.assert_equal(chol.shape[0], chol.shape[1])

    val = np.square(np.diag(chol))
    return np.sum(np.log(val))


@jit
def cholesky_solve(chol, X):
    # ensure square matrix
    chex.assert_rank(chol, 2)
    chex.assert_equal(chol.shape[0], chol.shape[1])

    # ensure shapes conform
    chex.assert_equal(chol.shape[1], X.shape[0])

    # assumes chol is lower
    return jax.scipy.linalg.cho_solve([chol, True], X)


@jit
def cholesky(A):
    # ensure square matrix
    chex.assert_rank(A, 2)
    chex.assert_equal(A.shape[0], A.shape[1])

    # return lower triangular cholesky factor
    return jax.scipy.linalg.cholesky(A, lower=True)
