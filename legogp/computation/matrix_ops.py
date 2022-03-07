"""Standard operations on matrices."""

import jax
import jax.numpy as np
from jax import jit
from functools import partial
import chex

@partial(jit, static_argnums=(1))
def get_block_diagonal(A, block_size):
    N = A.shape[0]

    num_blocks = N / block_size
    a = np.array(list(range(N)))

    indexes = np.reshape(a, [-1, block_size])

    blocks =  jax.vmap(
        lambda A, i: A[i,:][:, i],
        [None, 0],
        0
    )(A, indexes)

    chex.assert_shape(blocks, [num_blocks, block_size, block_size])

    return blocks

@jit
def cartesian_product(X, Y):
    return np.vstack([np.tile(X, Y.shape[0]), np.repeat(Y, X.shape[0])])

def add_jitter(A, jit):
    return A + jit*np.eye(A.shape[0])

@jit
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
def lower_triangular_cholesky(A):
    N = A.shape[0]
    return cholesky(A)[np.tril_indices(N)].flatten()

@jit
def vectorized_lower_triangular_cholesky(A:np.ndarray) -> np.ndarray:
    """
        Takes the cholesky decomposition of A vectorized the output
    """
    chex.assert_rank(A, 3)
    chex.assert_equal(A.shape[1], A.shape[2])

    A_chol_flattened = jax.vmap(
        lower_triangular_cholesky,
        [0],
        0
    )(A)

    return A_chol_flattened

@partial(jit, static_argnums=(1,))
def lower_triangle(val, N):
    tri = np.zeros((N, N))
    return jax.ops.index_update(tri, jax.ops.index[np.tril_indices(N, 0)], val)



@partial(jit, static_argnums=(1,))
def vectorized_lower_triangular(val:np.ndarray, N) -> np.ndarray:
    return jax.vmap(
        lambda x: lower_triangle(x, N),
        [0],
        0
    )(val)

@jit
def vec_columns(A):
    return A.T.reshape(A.shape[0]*A.shape[1], 1)
