import jax
import jax.numpy as np
import chex
import objax

from ..batching import loop_or_batch
from ..transforms import Independent, LinearTransform

from typing import List

def get_block_diag_diagional_gram_matrix(X: np.ndarray, kernels: List['Kernel']) -> np.ndarray:
    #assume all kernels are the same to support batching

    N = X.shape[0]
    num_latents = len(kernels)

    def _compute_gram_matrix(X, kernel):
        return kernel.K_diag(X)

    gram_arr = loop_or_batch(
        _compute_gram_matrix,
        [ X,  kernels ],
        [ None,  0 ],
        num_latents,
        num_returned_args=1
    )

    chex.assert_equal(gram_arr.shape[0], num_latents)
    chex.assert_equal(gram_arr.shape[1], N)

    return gram_arr


def get_block_diag_gram_matrix(X1: np.ndarray, X2: np.ndarray, kernels: List['Kernel']) -> np.ndarray:
    #assume all kernels are the same to support batching

    num_latents = len(kernels)

    def _compute_gram_matrix(X1, X2, kernel):
        return kernel.K(X1, X2)

    gram_arr = loop_or_batch(
        _compute_gram_matrix,
        [ X1, X2, kernels ],
        [ None, None, 0 ],
        num_latents,
        num_returned_args=1
    )

    gram_matrix = jax.scipy.linalg.block_diag(*gram_arr)

    N1 = X1.shape[0]
    N2 = X2.shape[0]
    chex.assert_equal(gram_matrix.shape[0], N1 * num_latents)
    chex.assert_equal(gram_matrix.shape[1], N2 * num_latents)

    return gram_matrix

def get_diagonal_gaussian_likelihood_variances(Y: np.ndarray, likelihood) -> np.ndarray:
    num_latents = Y.shape[1]
    N = Y.shape[0]

    def _compute_lik_variance(N, likelihood):
        return likelihood.variance * np.eye(N)

    var_arr = loop_or_batch(
        _compute_lik_variance,
        [ N, likelihood ],
        [ None, 0 ],
        num_latents,
        num_returned_args=1
    )

    var_arr = jax.scipy.linalg.block_diag(*var_arr)

    chex.assert_equal(var_arr.shape[0], N * num_latents)
    chex.assert_equal(var_arr.shape[1], N * num_latents)

    return var_arr

def get_vec_gaussian_likelihood_variances(Y: np.ndarray, likelihood) -> np.ndarray:
    num_latents = Y.shape[1]
    N = Y.shape[0]

    def _compute_lik_variance(N, likelihood):
        return likelihood.variance * np.ones(N)

    var_arr = loop_or_batch(
        _compute_lik_variance,
        [ N, likelihood ],
        [ None, 0 ],
        num_latents,
        num_returned_args=1
    )

    return var_arr

def get_linear_multi_task_prior_covariance(X1, X2, prior):
    mixing_matrix = prior.W
    kernels = prior.get_kernels()
    N1 = X1.shape[0]
    N2 = X2.shape[0]

    K_bdiag = get_block_diag_gram_matrix(X1, X2, kernels)

    W1 = np.kron(mixing_matrix, np.eye(N1))
    W2 = np.kron(mixing_matrix, np.eye(N2))

    covar = W1 @ K_bdiag @ W2.T

    return covar

def get_linear_multi_task_prior_diag_covariance(X, prior):
    mixing_matrix = prior.W
    kernels = prior.get_kernels()
    N = X.shape[0]

    # Q x N
    K_diag = get_block_diag_diagional_gram_matrix(X, kernels)

    W = mixing_matrix**2

    #P x N
    covar = W @ K_diag

    # PN
    covar = np.hstack(covar)

    return covar

def get_linear_multi_task_model_covariance(X: np.ndarray, Y: np.ndarray, likelihood: objax.ModuleList, prior: LinearTransform):
    """ Computes p(Y) = N(Y | 0, (W \kron I) K_bdiag (W \kron I)^T + eps). """

    covar = get_linear_multi_task_prior_covariance(X, X, prior)
    lik_diag = get_diagonal_gaussian_likelihood_variances(Y, likelihood)

    sigma = covar + lik_diag 

    return sigma


