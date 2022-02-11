from ...settings import jitter
from ...kernels import Kernel, RBF
from ...likelihood import Gaussian, GaussianParameterised, ProductLikelihood
from ...approximate_posteriors import GaussianApproximatePosterior, MeanFieldApproximatePosterior
from ...dispatch import dispatch, evoke
from ..gaussian import log_gaussian
from ...batching import loop_or_batch
from ...transforms import Independent, LinearTransform
from ..elbos import precompute_variational_primitives, precompute_diagonal_variational_primitives
from ..marginals import gaussian_conditional_diagional, gaussian_conditional_covar

from ...utils import utils
from ...utils.utils import can_batch, get_batch_type
from ...utils.batch_utils import batch_over_likelihoods
from ...utils.nan_utils import mask_to_identity, get_mask, mask_vector

from ..matrix_ops import cholesky, log_chol_matrix_det, add_jitter, cholesky_solve, vec_columns
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

@dispatch('BatchGP', Gaussian)
def predict_diagonal(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs):
    return  gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, likelihood.variance)

@dispatch('BatchGP', Gaussian)
def predict_covar(XS_1, XS_2, X, Y, likelihood, K_xs, K_xs_x, K_xx, K_x_xs, mean_x, mean_xs):

    return  gaussian_predictive_covar(Y, K_xs, K_xs_x, K_xx, K_x_xs, mean_x, mean_xs, likelihood.variance)


@dispatch(object, object, object, GaussianParameterised, object)
def predict_diagonal(XS, X, Y, likelihood, kernel):

    Ns = XS.shape[0]
    N = X.shape[0]

    K_xs = kernel.K_diag(XS)
    K_xs_x = kernel.K(XS, X)
    K_xx = kernel.K(X, X)

    lik_var = likelihood.variance(X)


    if (mask is not None):
        Y = np.nan_to_num(Y, nan=0.0)

        mask_xs_x = np.tile(mask, [XS.shape[0], 1]) 
        K_xs_x = np.multiply(K_xs_x, mask_xs_x)

        mask_xx = np.tile(mask, [mask.shape[0], 1]) 

        K_xx = K_xx-np.eye(N)
        K_xx = np.multiply(K_xx, mask_xx)
        K_xx = np.multiply(K_xx, mask_xx.T)
        K_xx = K_xx+np.eye(N)

        lik_var = lik_var-np.eye(N)
        lik_var = np.multiply(lik_var, mask_xx)
        lik_var = np.multiply(lik_var, mask_xx.T)
        lik_var = lik_var+np.eye(N)

    return  full_gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, lik_var)

@dispatch(object, object, GaussianApproximatePosterior, Gaussian, object, object)
def predict_diagonal(XS, X,  approximate_posterior, likelihood, kernel, sparsity):
    m, S_diag = approximate_posterior.predictive_marginal(XS, X, kernel, sparsity, diagonal=True)
    return m, S_diag + likelihood.variance

@dispatch(object, object, GaussianApproximatePosterior, Gaussian, object, object)
def predict_full(XS, X,  approximate_posterior, likelihood, kernel, sparsity):
    m, S = approximate_posterior.predictive_marginal(XS, X, kernel, sparsity, diagonal=False)
    return m, S + np.eye(XS.shape[0])*likelihood.variance


@dispatch(Independent, MeanFieldApproximatePosterior)
def multi_latent_predict(XS, X, Y, likelihood, prior, approximate_posterior, diagonal):
    mean_xx_arr, mean_zz_arr, K_x_arr, K_xz_arr, K_zz_arr, m_arr, S_chol_arr, S_arr = precompute_diagonal_variational_primitives(XS, prior, approximate_posterior)

    P = len(approximate_posterior.approx_posteriors)

    mean_x = np.tile(
        np.zeros([m_arr.shape[1], 1]), [P, 1, 1]
    )
    mean_xs = np.tile(
        np.zeros([XS.shape[0], 1]), [P, 1, 1]
    )

    mu, sig = batch_or_loop(
        gaussian_conditional_diagional,
        [XS, X, K_zz_arr, K_xz_arr, K_x_arr, m_arr, S_chol_arr, mean_x, mean_xs],
        [None, None, 0, 0, 0, 0, 0, 0, 0],
        dim=P,
        out_dim=2,
        batch_type = BatchType.LOOP
    )

    return mu, sig

@dispatch(Independent, MeanFieldApproximatePosterior)
def multi_latent_predictive_covar(X1, X2, X, Y, likelihood, prior, approximate_posterior):

    mean_xx_arr, mean_zz_arr, K_x_arr, K_xz_arr, K_zz_arr, m_arr, S_chol_arr, S_arr = precompute_variational_primitives(X1, prior, approximate_posterior)

    K_x_arr = prior.covar(X1, X2)


    K_zx_arr = batch_or_loop(
        lambda prior: prior.kernel.K(prior.sparsity.Z, X2),
        [prior.latents],
        [0],
        dim = len(prior.latents),
        out_dim = 1,
        batch_type = BatchType.LOOP
    )

    P = len(approximate_posterior.approx_posteriors)

    sig = batch_or_loop(
        gaussian_conditional_covar,
        [X1, X2, X, K_zz_arr, K_xz_arr, K_zx_arr, K_x_arr, m_arr, S_chol_arr],
        [None, None, None, 0, 0, 0, 0, 0, 0],
        dim=P,
        out_dim=1,
        batch_type = BatchType.LOOP
    )

    chex.assert_shape(sig, [P, X1.shape[0], X2.shape[0]])
    return sig

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

    mu_arr, var_arr =  batch_over_likelihoods(
        evoke_name,
        [gp],
        likelihood_arr,
        [XS, X, Y, likelihood_arr, K_xs, K_xs_x, K_xx, mean_x, mean_xs],
        [None, None, 1, 0, 0, 0, 0, 0, 0],
        num_latents,
        2
    )

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

    var_arr =  batch_over_likelihoods(
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
