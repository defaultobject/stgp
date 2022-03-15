from ...settings import jitter
from ...kernels import Kernel, RBF
from ...likelihood import Gaussian, GaussianParameterised, ProductLikelihood
from ...approximate_posteriors import GaussianApproximatePosterior, MeanFieldApproximatePosterior
from ...dispatch import dispatch, evoke
from ..gaussian import log_gaussian
from ...transforms import Independent, LinearTransform

from ...utils import utils
from ...utils.utils import can_batch, get_batch_type
from ...utils.nan_utils import mask_to_identity, get_mask, mask_vector, get_diag_mask

from ..matrix_ops import cholesky, log_chol_matrix_det, add_jitter, cholesky_solve, vec_columns, triangular_solve, block_diagonal_from_cholesky, block_from_vec, v_get_block_diagonal, block_from_mat

import jax
from jax import jit
from functools import partial
import jax.numpy as np
import chex
from typing import List
from objax import ModuleList
from batchjax import batch_or_loop, BatchType

@jit
def gaussian_predictive_mean(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
    mask = get_mask(Y)
    M = get_diag_mask(mask)
    Y = mask_vector(Y, mask)

    Ns = K_xs.shape[0]
    N = Y.shape[0]

    k = M @ K_xx @ M + lik_var
    k_chol = cholesky(k)

    mu = K_xs_x @ M @ cholesky_solve(k_chol, Y-mean_x) + mean_xs
    mu = np.reshape(mu, [Ns, 1])

    return mu

@jit
def gaussian_predictive_covar(Y, K_xs, K_xs_x, K_xx, K_x_xs, mean_x, mean_xs, lik_var):
    mask = get_mask(Y)
    M = get_diag_mask(mask)

    Ns_1 = K_xs.shape[0]
    Ns_2 = K_xs.shape[1]
    N = Y.shape[0]

    k = M @ K_xx @ M + lik_var
    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol, M @ K_xs_x.T, lower=True)
    A2 = jax.scipy.linalg.solve_triangular(k_chol, M @ K_x_xs, lower=True)
    sig = K_xs - A1.T @ A2

    sig = np.reshape(sig, [Ns_1, Ns_2])

    return sig


@jit
def gaussian_prediction(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
    mask = get_mask(Y)
    M = get_diag_mask(mask)
    Y = mask_vector(Y, mask)

    Ns = K_xs.shape[0]

    k = M @ K_xx @ M + lik_var

    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol, M @ K_xs_x.T, lower=True)

    mu = K_xs_x @ M @ cholesky_solve(k_chol, Y-mean_x) + mean_xs
    sig = K_xs - A1.T @ A1

    mu = np.reshape(mu, [Ns, 1])
    sig = np.reshape(sig, [Ns, Ns])


    return mu, sig


@jit 
def gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
    mask = get_mask(Y)
    M = get_diag_mask(mask)
    Y = mask_vector(Y, mask)

    k = M.T @ K_xx @ M + lik_var

    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol, M @ K_xs_x.T, lower=True)

    mu = K_xs_x  @ M @ cholesky_solve(k_chol, Y-mean_x) + mean_xs

    sig = K_xs - np.sum(np.square(A1), axis=0)
    sig = sig[:, None]

    chex.assert_equal(mu.shape, sig.shape)

    #K_xs - np.diag(K_xs_x @ cholesky_solve(k_chol, K_xs_x.T))
    #K_xs_x @ cholesky_solve(k_chol, Y)

    #np.sum(np.abs((K_xs - np.diag(K_xs_x @ cholesky_solve(k_chol, K_xs_x.T)))-sig[:, 0]))
    #np.sum(np.abs(K_xs_x @ cholesky_solve(k_chol, Y)-mu))

    return mu, sig

@partial(jit, static_argnums=(0))
def gaussian_prediction_blocks(group_size, block_size, Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
    chex.assert_shape(K_xx, lik_var.shape)

    # TODO: add means and missing data
    K = K_xx + lik_var
    K_chol = cholesky(K)

    #K_xs[0] - K_xs_x @ cholesky_solve(K_chol, K_xs_x.T)
    A = triangular_solve(
        K_chol, 
        K_xs_x.T, 
        lower=True
    )

    B = block_diagonal_from_cholesky(A.T, block_size)

    sig = K_xs - B

    # K_xs_x @ cholesky_solve(K_chol, Y)
    mu = triangular_solve(K_chol.T, A, lower=False).T @ Y
    mu = block_from_vec(mu, block_size)

    return mu, sig
