from ...settings import jitter
from ...kernels import Kernel, RBF
from ...likelihood import Gaussian, GaussianParameterised, ProductLikelihood, DiagonalGaussian, Likelihood, BlockDiagonalGaussian
from ...approximate_posteriors import GaussianApproximatePosterior, MeanFieldApproximatePosterior, ApproximatePosterior
from ...dispatch import dispatch, evoke
from ..gaussian import log_gaussian
from ...batching import loop_or_batch
from ...transforms import Independent, Transform, LinearTransform, NonLinearTransform
from ..marginals import gaussian_conditional_diagional, gaussian_conditional_covar

from ...utils import utils
from ...utils.utils import can_batch, get_batch_type
from ...utils.batch_utils import batch_over_module_types
from ...utils.nan_utils import mask_to_identity, get_mask, mask_vector

from ..matrix_ops import cholesky, log_chol_matrix_det, add_jitter, cholesky_solve, vec_columns, get_block_diagonal
from ..model_ops import get_block_diag_gram_matrix, get_diagonal_gaussian_likelihood_variances, get_linear_multi_task_model_covariance, get_linear_multi_task_prior_covariance, get_linear_multi_task_prior_diag_covariance

from .base_predictors import gaussian_prediction, gaussian_predictive_covar, gaussian_predictive_mean, gaussian_prediction_diagonal

import jax
from jax import jit
import jax.numpy as np
import chex
from typing import List
from objax import ModuleList
from batchjax import batch_or_loop, BatchType


@dispatch('BatchGP', Gaussian)
def predict(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs):
    chex.assert_equal(K_xx.shape, (X.shape[0], X.shape[0]))
    chex.assert_equal(K_xs_x.shape, (XS.shape[0], X.shape[0]))
    chex.assert_equal(K_xs.shape, (XS.shape[0], XS.shape[0]))

    return gaussian_prediction(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, likelihood.variance)

@dispatch('BatchGP', DiagonalGaussian)
def predict_diagonal(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs):
    # Supports likelihood.variance being scalar or a vector
    return  gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, likelihood.variance)

@dispatch('BatchGP', Gaussian)
def predict_covar(XS_1, XS_2, X, Y, likelihood, K_xs, K_xs_x, K_xx, K_x_xs, mean_x, mean_xs):
    return  gaussian_predictive_covar(Y, K_xs, K_xs_x, K_xx, K_x_xs, mean_x, mean_xs, likelihood.variance)


@dispatch('BatchGP', ProductLikelihood, Independent)
def predict(XS, X, Y, gp, likelihood, prior, diagonal: bool):
    num_latents = prior.num_latents
    num_outputs = prior.num_outputs

    if diagonal:
        K_xs = prior.var(XS)
        evoke_name = 'predict_diagonal'
    else:
        K_xs = prior.covar(XS, XS)
        evoke_name = 'predict'

    # Precompute batched kernels
    K_xx = prior.covar(X, X)
    K_xs_x = prior.covar(XS, X)

    mean_x = prior.mean(X)
    mean_xs = prior.mean(XS)

    likelihood_arr = likelihood.likelihood_arr

    # Ensure Y is rank 2 after batching
    Y = Y[..., None]

    mu_arr, var_arr =  batch_over_module_types(
        evoke_name,
        [gp],
        likelihood_arr,
        [XS, X, Y, likelihood_arr, K_xs, K_xs_x, K_xx, mean_x, mean_xs],
        [None, None, 1, 0, 0, 0, 0, 0, 0],
        num_latents,
        2
    )

    Ns = XS.shape[0]
    P = Y.shape[1]

    mu_arr = mu_arr.reshape([P, Ns])
    var_arr = var_arr.reshape([P, Ns])

    return mu_arr, var_arr

@dispatch('BatchGP', ProductLikelihood, Independent)
def predict_covar(XS_1, XS_2, X, Y, gp, likelihood, prior):
    num_latents = prior.num_latents
    num_outputs = prior.num_outputs

    # precompute batched kernels
    
    K_xs = prior.covar(XS_1, XS_2)
    K_xx = prior.covar(X, X)
    K_xs_x = prior.covar(XS_1, X)
    K_x_xs = prior.covar(X, XS_2)
    mean_x = prior.mean(X)
    mean_xs_1 = prior.mean(XS_1)
    mean_xs_2 = prior.mean(XS_2)

    likelihood_arr = likelihood.likelihood_arr

    # Ensure Y is rank 2 after batching
    Y = Y[..., None]

    var_arr =  batch_over_module_types(
        'predict_covar',
        [gp],
        likelihood_arr,
        [XS_1, XS_2, X, Y, likelihood_arr, K_xs, K_xs_x, K_xx, K_x_xs, mean_x, mean_xs_1],
        [None, None, None, 1, 0, 0, 0, 0, 0, 0, 0],
        num_latents,
        1
    )

    chex.assert_shape(var_arr, [num_outputs, XS_1.shape[0], XS_2.shape[0]])

    return var_arr


@dispatch('BatchGP', ProductLikelihood, LinearTransform)
def predict(XS, X, Y, gp, likelihood, prior, diagonal):
    Ns = XS.shape[0]
    N = X.shape[0]
    P = Y.shape[1]

    likelihood_arr = likelihood.likelihood_arr

    K_xs = prior.vec_var(XS)[:, 0]
    K_xx = prior.full_covar(X, X)
    K_xs_x = prior.full_covar(XS, X)
    lik_var = get_diagonal_gaussian_likelihood_variances(Y, likelihood_arr)
    mean_x = prior.vec_mean(X)
    mean_xs = prior.vec_mean(XS)

    Y_vec = vec_columns(Y)

    # gaussian_prediction(_*) support both lik_var being a scalar and a diagonal matrix
    if diagonal:
        mu, var = gaussian_prediction_diagonal(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)
    else:
        mu, var = gaussian_prediction(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

    mu = mu.reshape([P, Ns])
    var = var.reshape([P, Ns])

    return mu, var

@dispatch('BatchGP', BlockDiagonalGaussian, LinearTransform)
def predict(XS, X, Y, gp, likelihood, prior, diagonal):
    Ns = XS.shape[0]
    N = X.shape[0]
    P = Y.shape[1]

    K_xs = prior.vec_var(XS)[:, 0]
    K_xx = prior.full_covar(X, X)
    K_xs_x = prior.full_covar(XS, X)

    # Get liklihood
    likelihood_var = jax.scipy.linalg.block_diag(*likelihood.variance)

    # Permute so that the ordering between likelihood_var and Y is the same
    N = X.shape[0]
    NS = likelihood_var.shape[0]

    i = np.hstack([np.arange(i , NS, P) for i in range(P)])
    permutaton = np.eye(NS)[i]

    lik_var = permutaton @ likelihood_var @ permutaton.T

    mean_x = prior.vec_mean(X)
    mean_xs = prior.vec_mean(XS)

    Y_vec = vec_columns(Y)

    # gaussian_prediction(_*) support both lik_var being a scalar and a  matrix
    if diagonal:
        mu, var = gaussian_prediction_diagonal(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)
    else:
        mu, var = gaussian_prediction(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

    mu = mu.reshape([P, Ns])
    var = var.reshape([P, Ns])

    return mu, var

@dispatch('BatchGP', BlockDiagonalGaussian, LinearTransform)
def predict_blocks(XS, X, Y, gp, likelihood, prior, diagonal):

    Ns = XS.shape[0]
    N = X.shape[0]
    P = Y.shape[1]


    K_xs = prior.vec_var(XS)[:, 0]
    K_xx = prior.full_covar(X, X)
    K_xs_x = prior.full_covar(XS, X)

    # Get liklihood
    likelihood_var = jax.scipy.linalg.block_diag(*likelihood.variance)

    # Permute so that the ordering between likelihood_var and Y is the same
    N = X.shape[0]
    NS = likelihood_var.shape[0]

    i = np.hstack([np.arange(i , NS, P) for i in range(P)])
    permutaton = np.eye(NS)[i]

    lik_var = permutaton @ likelihood_var @ permutaton.T

    mean_x = prior.vec_mean(X)
    mean_xs = prior.vec_mean(XS)

    Y_vec = vec_columns(Y)

    # TODO: this is v. inefficient
    # Compute full matrix
    mu, var = gaussian_prediction(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

    K = permutaton @ var @ permutaton.T

    var = get_block_diagonal(K, likelihood.block_size)

    mu = mu.reshape([P, Ns]).T

    return mu, var


@dispatch(Likelihood, Transform, ApproximatePosterior)
def predict(XS, X, Y, likelihood, prior, approximate_posterior, inference, diagonal):

    return  evoke('marginal', 'prediction', approximate_posterior, prior)(
        XS, X, approximate_posterior, prior, inference
    )
