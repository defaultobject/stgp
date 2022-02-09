"""Standard operations on matrices."""

import jax
import jax.numpy as np
from jax import jit
from functools import partial
import chex

@jit
def cartesian_product(X, Y):
    return np.vstack([np.tile(X, Y.shape[0]), np.repeat(Y, X.shape[0])])

def add_jitter(A, jit):
    return A + jit*np.eye(A.shape[0])

def diagonal_from_cholesky(L):
    """ Compute diag(LL^T). """
    # ensure square matrix
    chex.assert_rank(L, 2)
    chex.assert_equal(L.shape[0], L.shape[1])

    diag = np.sum(np.square(L), axis=1)
    diag = np.reshape(diag, [L.shape[0], 1])

    return diag

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

@partial(jit, static_argnums=(2))
def _triangular_solve(chol, X, lower):
    return jax.scipy.linalg.solve_triangular(chol, X, lower=lower) 

def triangular_solve(chol, X, lower):
    #wrapper around _triangular_solve so that lower can be a keyword arg
    return _triangular_solve(chol, X, lower)


@jit
def vectorized_lower_triangular_cholesky(A:np.ndarray) -> np.ndarray:
    """
        Takes the cholesky decomposition of A vectorized the output
    """
    N = A.shape[0]
    #init = cholesky(A)+1e-7*np.eye(N) #add jitter for numerical stability
    init = cholesky(A)
    init = init[np.tril_indices(N)].flatten()
    return init

@partial(jit, static_argnums=(1,))
def lower_triangle(val, N):
    tri = np.zeros((N, N))
    return jax.ops.index_update(tri, jax.ops.index[np.tril_indices(N, 0)], val)


@jit
def vec_columns(A):
    return A.T.reshape(A.shape[0]*A.shape[1], 1)
