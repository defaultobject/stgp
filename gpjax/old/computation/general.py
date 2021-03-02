from jax.config import config
import jax
import jax.numpy as np
from jax import jit, partial
from jax.ops import index, index_add
from jax.experimental import loops
from jax.scipy.special import erf, gammaln

from ..settings import Settings


# TODO: fix imports
# from .kalman_filter import kalman_loop, kalman_loop_store_intermediate
# from .rts_smoother import rts_smoother


@jit
def log_poisson(x, lam):
    return x * np.log(lam) - lam - gammaln(x + 1.0)


@jit
def inv_probit(f):
    """
    See gpflow implementation:
        https://github.com/GPflow/GPflow/blob/282d1cdd1cea79702aeced3781239e79c2197f7c/gpflow/likelihoods/utils.py#L19
    """
    jitter = 1e-3
    return 0.5 * (1.0 + erf(f / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter


@jit
def inv_logit(f):
    return 1 / (1 + np.exp(-f))


@jit
def log_chol_matrix_det(chol):
    val = np.square(np.diag(chol))
    return np.sum(np.log(val))


@jit
def cholesky(A):
    # return lower triangular cholesky factor
    return jax.scipy.linalg.cholesky(A, lower=True)


@partial(jit, static_argnums=(2))
def _triangular_solve(chol, X, lower):
    return jax.scipy.linalg.solve_triangular(chol, X, lower=lower)


def triangular_solve(chol, X, lower):
    # wrapper around _triangular_solve so that lower can be a keyword arg
    return _triangular_solve(chol, X, lower)


@jit
def cholesky_solve(chol, X):
    # assumes chol is lower
    # return jax.scipy.linalg.solve_triangular( chol.T, jax.scipy.linalg.solve_triangular(chol, X, lower=True))
    return jax.scipy.linalg.cho_solve([chol, True], X)


@jit
def positive_transform(val):
    # return np.exp(val)
    return np.log(1 + np.exp(val))
    # return np.logaddexp(val, 0.)


@jit
def negative_transform(val):
    return -positive_transform(val)
    # return np.logaddexp(val, 0.)


@jit
def inv_positive_transform(val):
    # return np.log(val)
    return np.log(np.exp(val) - 1)


@partial(jit, static_argnums=(1,))
def lower_triangle(val, N):
    tri = np.zeros((N, N))
    return jax.ops.index_update(tri, jax.ops.index[np.tril_indices(N, 0)], val)


@partial(jit, static_argnums=(1,))
def flatten_cholesky(val, N):
    tri = np.zeros((N, N))
    return val[np.tril_indices(N, 0)]


@jit
def sigmoid(val):
    return 1 / (1 + np.exp(-val))


@jit
def inv_sigmoid(val):
    return -np.log((1 / val) - 1)


@jit
def correlation_transform(val, a):
    return 2 * sigmoid(a * val) - 1


@jit
def inv_correlation_transform(val, a):
    return inv_sigmoid((val + 1) / 2) / a


@partial(jit, static_argnums=(1, 2, 5))
def get_computational_primitives(Y, N, dt, kernel, likelihood, mask):
    (
        neg_log_marg_lik,
        filtered_mean,
        filtered_cov,
        alpha,
        beta,
        chol_diag,
    ) = kalman_loop_store_intermediate(Y, N, dt, kernel, likelihood, mask)

    mu, sig, alpha = rts_smoother(
        Y, N, dt, filtered_mean, filtered_cov, kernel, likelihood, mask, alpha
    )

    return alpha, beta, chol_diag


@partial(jit, static_argnums=(1, 2, 5))
def kf_cholesky_solve_trace(L, N, dt, kernel, likelihood, mask):
    with loops.Scope() as s:
        s.kf = 0.0
        for i in s.range(N):
            L_col = L[:, i][:, None]
            _, _, _, alpha_i, beta_i, _ = kalman_loop(
                L_col, N, dt, kernel, likelihood, mask
            )
            s.kf += beta_i

        return s.kf


@partial(jit, static_argnums=(1, 2))
def get_correlation_cholesky(z_arr, P, Q):
    """
    Constructs a correlation matrix given z_arr and returns the cholesky
    Args:
        z_arr: Q - array of real numbers
    Returns
        P x P cholesky of correlation matrix
    """
    # Identity matrix
    # [ 1 0 ]
    # [ 0 1 ]
    chol = np.eye(P)

    # identity + raw z's in lower triangle
    # [ 1 0 ]
    # [ z 1 ]
    chol = jax.ops.index_add(chol, jax.ops.index[np.tril_indices(P, -1)], z_arr)

    # construct elements (1-z**2)^0.5
    # [ 0 1 ]
    # [ (1-z^2)^0.5 0 ]
    chol_a = (1 - chol ** 2) ** 0.5

    # zero out elements in upper triangle
    # [ 0 0 ]
    # [ (1-z^2)^0.5 0 ]
    upper_tri_index = np.triu_indices(P, 1)
    chol_a = jax.ops.index_add(
        chol_a, jax.ops.index[upper_tri_index], -chol_a[upper_tri_index]
    )

    # Add ones back onto the diagional
    # [ 1 0 ]
    # [ (1-z^2)^0.5 1 ]
    chol_a = chol_a + np.eye(P)

    # Calculate accumulative product across rows
    # [ 1 1 ]
    # [ 1 (1-z^2)^0.5 ]
    def fn(carry, y):
        t = np.multiply(carry, y)
        return t, t

    _, chol_b = jax.lax.scan(fn, np.ones(P), chol_a.T)
    chol_a = chol_b.T
    chol_a = np.concatenate([np.ones([P, 1]), chol_a[:, :-1]], axis=1)

    # Add original z back to finish cholesky
    # [ 1 1 ]
    # [ z (1-z^2)^0.5 ]
    chol = np.multiply(chol, chol_a)

    # construct full correlation matrix
    weights = chol @ chol.T

    return chol


@partial(jit, static_argnums=(1, 2))
def get_z_from_correlation_cholesky(R_chol, P, Q):
    # first column has raws z's
    # [ 1 0 0 ]
    # [z1 ? 0 ]
    # [z2 ? ? ]

    chol = np.eye(P)

    # raw z's in first col
    # [ 2 0 0]
    # [ z1 1 0]
    # [ z2 0 1]
    chol = jax.ops.index_add(chol, jax.ops.index[:, 0], R_chol[:, 0])
    chol = jax.ops.index_add(chol, jax.ops.index[0, 0], -1)

    # construct elements (1-z**2)^0.5
    # [ 0 1 ]
    # [ (1-z^2)^0.5 0 ]
    chol_a = (1 - chol ** 2) ** 0.5

    # zero out elements in upper triangle
    # [ 0 0 ]
    # [ (1-z^2)^0.5 0 ]
    upper_tri_index = np.triu_indices(P, 1)
    # chol_a = jax.ops.index_add(chol_a, jax.ops.index[upper_tri_index], -chol_a[upper_tri_index])

    # Add ones back onto the diagional
    # [ 1 0 ]
    # [ (1-z^2)^0.5 1 ]
    chol_a = chol_a + np.eye(P)

    # Calculate accumulative product across rows
    # [ 1 1 ]
    # [ 1 (1-z^2)^0.5 ]
    def fn(carry, y):
        t = np.multiply(carry, y)
        return t, t

    _, chol_b = jax.lax.scan(fn, np.ones(P), chol_a.T)
    chol_a = chol_b.T
    chol_a = np.concatenate([np.ones([P, 1]), chol_a[:, :-1]], axis=1)

    # Find z's on on other columns and add original z's back
    # [ 1 0 0 ]
    # [z1 ? 0 ]
    # [z2 z3 ? ]
    chol_a = np.multiply(R_chol, 1 / chol_a)

    # index into lower diagional and return z
    z = chol_a[np.tril_indices(P, -1)]
    return z
