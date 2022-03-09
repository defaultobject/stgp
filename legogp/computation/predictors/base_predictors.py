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

from ..matrix_ops import cholesky, log_chol_matrix_det, add_jitter, cholesky_solve, vec_columns, triangular_solve

import jax
from jax import jit
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

    k = add_jitter(M @ K_xx @ M, lik_var)
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

    k = add_jitter(M @ K_xx @ M , lik_var)
    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol, M @ K_xs_x.T, lower=True)
    A2 = jax.scipy.linalg.solve_triangular(k_chol, M @ K_x_xs, lower=True)
    sig = K_xs - A1.T @ A2

    sig = np.reshape(sig, [Ns_1, Ns_2])

    return sig

@jit
def _gaussian_prediction_no_nans(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
    Ns = K_xs.shape[0]

    k = add_jitter(K_xx, lik_var)
    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol, K_xs_x.T, lower=True)

    mu = K_xs_x @ cholesky_solve(k_chol, Y-mean_x) + mean_xs
    sig = K_xs - A1.T @ A1

    mu = np.reshape(mu, [Ns, 1])
    sig = np.reshape(sig, [Ns, Ns])

    return mu, sig

@jit
def gaussian_prediction(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
    mask = get_mask(Y)
    M = get_diag_mask(mask)
    Y = mask_vector(Y, mask)

    Ns = K_xs.shape[0]

    k = add_jitter(M @ K_xx @ M, lik_var)
    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol, M @ K_xs_x.T, lower=True)

    mu = K_xs_x @ M @ cholesky_solve(k_chol, Y-mean_x) + mean_xs
    sig = K_xs - A1.T @ A1

    mu = np.reshape(mu, [Ns, 1])
    sig = np.reshape(sig, [Ns, Ns])

    return mu, sig

@jit 
def _gaussian_prediction_diagonal_no_nans(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
    """ Diagonal Gaussian prediction without masking """
    k = add_jitter(K_xx , lik_var)

    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol,  K_xs_x.T, lower=True)

    mu = K_xs_x  @ cholesky_solve(k_chol, Y-mean_x) + mean_xs

    sig = K_xs - np.sum(np.square(A1), axis=0)
    sig = sig[:, None]

    chex.assert_equal(mu.shape, sig.shape)

    return mu, sig

@jit 
def gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
    mask = get_mask(Y)
    M = get_diag_mask(mask)
    Y = mask_vector(Y, mask)

    k = add_jitter(M.T @ K_xx @ M, lik_var)

    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol, M @ K_xs_x.T, lower=True)

    mu = K_xs_x  @ M @ cholesky_solve(k_chol, Y-mean_x) + mean_xs

    sig = K_xs - np.sum(np.square(A1), axis=0)
    sig = sig[:, None]

    chex.assert_equal(mu.shape, sig.shape)

    return mu, sig

@jit
def gaussian_prediction_blocks(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
    # TODO: add means and missing data
    block_size = K_xs.shape[1]

    K = K_xx + lik_var
    K_chol = cholesky(K)

    A = triangular_solve(
        K_chol, 
        K_xs_x.T, 
        lower=True
    )

    A1 = np.reshape(A, [-1, block_size, A.shape[0]])
    B = A1 @ np.transpose(A1, [0, 2, 1])
    sig = K_xs - B

    mu = triangular_solve(K_chol.T, A, lower=False).T @ Y
    mu = np.reshape(mu, [-1, block_size])

    return mu, sig
