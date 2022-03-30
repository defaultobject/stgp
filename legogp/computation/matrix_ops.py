"""Standard operations on matrices."""

import jax
import jax.numpy as np
from jax import jit
from functools import partial
import chex

def to_block_diag(A):
    return jax.scipy.linalg.block_diag(*A)

@partial(jit, static_argnums=(1))
def get_block_diagonal(A, block_size):
    chex.assert_rank(A, 2)

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

@partial(jit, static_argnums=(1))
def batched_block_diagional(A, block_size):
    chex.assert_rank(A, 3)
    return jax.vmap(get_block_diagonal, [0, None])(A, block_size)

@jit
def cartesian_product(X, Y):
    return np.vstack([np.tile(X, Y.shape[0]), np.repeat(Y, X.shape[0])])

@jit
def add_jitter(A, jit):
    chex.assert_rank(A, 2)
    return A + jit*np.eye(A.shape[0])

@jit
def vec_add_jitter(A, jit):
    chex.assert_rank(A, 3)

    return jax.vmap(
        add_jitter,
        (0, None),
        0
    )(A, jit)

@jit
def diagonal_from_cholesky(L):
    """ Compute diag(LL^T). """
    # ensure square matrix
    chex.assert_rank(L, 2)
    chex.assert_equal(L.shape[0], L.shape[1])

    diag = np.sum(np.square(L), axis=1)
    diag = np.reshape(diag, [L.shape[0], 1])

    return diag


@partial(jit, static_argnums=(1))
def block_diagonal_from_cholesky(L, block_size):
    """
    Extracts block diagonals from L L^T
    """

    L1 = np.reshape(
        L, 
        [-1, block_size, L.shape[-1]]
    )

    B = L1 @ np.transpose(L1, [0, 2, 1])

    return B

@partial(jit, static_argnums=(1))
def block_from_vec(x, block_size):
    return np.reshape(x, [-1, block_size])

@partial(jit, static_argnums=(1))
def block_from_mat(X, block_size):
    return np.reshape(X, [-1, block_size, X.shape[1]])

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
    # return lower triangular cholesky factor
    return np.linalg.cholesky(A)

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
def stack_columns(A):
    """
    Stacks columns of A:

                    [0]
        [0, 1]  ->  [2]
        [2, 3]      [1]
                    [3]
    """ 
    chex.assert_rank(A, 2)
    return A.T.reshape(A.shape[0]*A.shape[1], 1)

vec_columns = stack_columns

@jit 
def stack_rows(A):
    """
    Stacks rows of A:

                    [0]
        [0, 1]  ->  [1]
        [2, 3]      [2]
                    [3]
    """ 
    chex.assert_rank(A, 2)
    return np.vstack(A[..., None])

@partial(jit, static_argnums=(1, 2))
def p_get_block_diagonal(A, b_size, A_dim):
    chex.assert_rank(A, 2)

    if b_size == 1:
        return np.diagonal(A)[:, None]

    if b_size == A_dim:
        return A

    raise NotImplementedError()

@partial(jit, static_argnums=(1, 2))
def v_get_block_diagonal(A, b_size, A_dim):
    return jax.vmap(
        p_get_block_diagonal, 
        (0, None, None),
        0
    )(A, b_size, A_dim)
