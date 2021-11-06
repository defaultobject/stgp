from ..settings import jitter
from ..kernels import Kernel, RBF
from ..likelihood import Gaussian, GaussianParameterised
from ..approximate_posteriors import GaussianApproximatePosterior
from ..dispatch import dispatch, evoke
from .gaussian import log_gaussian
from ..batching import loop_or_batch
from ..transforms import Independent, LinearTransform
from ..utils import utils
from ..utils.nan_utils import mask_to_identity, get_mask, mask_vector
from .matrix_ops import cholesky, log_chol_matrix_det, add_jitter, cholesky_solve
from .model_ops import get_block_diag_gram_matrix, get_diagonal_gaussian_likelihood_variances, get_linear_multi_task_model_covariance, get_linear_multi_task_prior_covariance, get_linear_multi_task_prior_diag_covariance

import jax
from jax import jit
import jax.numpy as np
import chex
from typing import List
from objax import ModuleList

@jit
def gaussian_prediction(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
    Ns = K_xs.shape[0]
    N = Y.shape[0]

    k = K_xx + np.eye(N)*lik_var

    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol, K_xs_x.T, lower=True)

    mu = K_xs_x @ cholesky_solve(k_chol, Y-mean_x) + mean_xs
    sig = K_xs - A1.T @ A1

    mu = np.reshape(mu, [Ns, 1])
    sig = np.reshape(sig, [Ns, Ns])

    return mu, sig

@jit 
def full_gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_xx):
    k = K_xx + lik_xx

    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol, K_xs_x.T, lower=True)

    mu = K_xs_x @ cholesky_solve(k_chol, Y-mean_x) + mean_xs
    sig = K_xs - np.sum(np.square(A1), axis=0)
    sig = sig[:, None]

    chex.assert_equal(mu.shape, sig.shape)

    return mu, sig



@jit 
def full_gaussian_predictive_covar(Y, K_xs, K_xs_x, K_x_xs, K_xx, lik_xx):
    k = K_xx + lik_xx
    k_chol = cholesky(k)

    sig = K_xs - K_xs_x @ cholesky_solve(k_chol, K_x_xs)

    return sig

@dispatch(object, object, object, object, object, Gaussian)
def full_predictive_covar(Y, K_xs, K_xs_x, K_x_xs, K_xx, likelihood):
    N = Y.shape[0]
    lik_xx = np.eye(N)*likelihood.variance

    return full_gaussian_predictive_covar(Y, K_xs, K_xs_x, K_x_xs, K_xx, lik_xx)

@jit
def gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
    N = Y.shape[0]
    lik_xx = np.eye(N)*lik_var

    return full_gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_xx)
    

@dispatch(object, object, object, Gaussian, object, object, object, object, object)
def predict(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs):

    Ns = XS.shape[0]
    N = X.shape[0]

    chex.assert_equal(K_xx.shape, (X.shape[0], X.shape[0]))
    chex.assert_equal(K_xs_x.shape, (XS.shape[0], X.shape[0]))
    chex.assert_equal(K_xs.shape, (XS.shape[0], XS.shape[0]))

    return gaussian_prediction(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, likelihood.variance)

@dispatch(object, object, object, Gaussian, object, object, object, object, object)
def predict_diagonal(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs):

    Ns = XS.shape[0]
    N = X.shape[0]

    mask = get_mask(Y)

    # TODO: document and test this
    Y = np.nan_to_num(Y, nan=0.0)


    mask_xs_x = np.tile(mask, [XS.shape[0], 1]) 
    K_xs_x = np.multiply(K_xs_x, mask_xs_x)

    K_xx = mask_to_identity(K_xx, mask)

    mean_x =  mask_vector(mean_x, mask)

    return  gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, likelihood.variance)

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

@dispatch(object, object, object, object, object, Independent)
def multi_latent_predictive_covar(XS_1, XS_2, X, Y, likelihood, prior):
    num_latents = prior.num_latents
    num_outputs = prior.num_outputs

    # Precompute batched kernels
    K_xs = prior.covar(XS_1, XS_2)
    K_xx = prior.covar(X, X)
    K_xs_x = prior.covar(XS_1, X)
    K_x_xs = prior.covar(X, XS_2)

    Y = Y[..., None]

    pred_fn = evoke('full_predictive_covar')

    var_arr = loop_or_batch(
        pred_fn,
        [Y, K_xs, K_xs_x, K_x_xs, K_xx, likelihood],
        [1, 0, 0, 0, 0, 0, 0],
        num_latents,
        num_returned_args=1
    )

    chex.assert_shape(var_arr, [num_outputs, XS_1.shape[0], XS_2.shape[0]])

    return var_arr

@dispatch(object, object, object, object, Independent, object)
def multi_latent_predict(XS, X, Y, likelihood, prior, diagonal):
    """ Independent Latent functions. Each predictions is computed separately"""

    num_latents = prior.num_latents
    num_outputs = prior.num_outputs

    if diagonal:
        pred_fn = evoke('predict_diagonal')

        # Precompute batched kernels
        K_xs = prior.var(XS)
        K_xx = prior.covar(X, X)
        K_xs_x = prior.covar(XS, X)
    else:
        pred_fn = evoke('predict')

        # Precompute batched kernels
        K_xs = prior.covar(XS, XS)
        K_xx = prior.covar(X, X)
        K_xs_x = prior.covar(XS, X)

    mean_x = prior.mean(X)
    mean_xs = prior.mean(XS)


    Y = Y[..., None]

    mu_arr, var_arr = loop_or_batch(
            lambda XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs: pred_fn(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x[:, None], mean_xs[:, None]),
        [XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs],
        [None, None, 1, 0, 0, 0, 0, 0, 0],
        num_latents,
        num_returned_args=2
    )

    return mu_arr, var_arr

@dispatch(object, object, object, object, LinearTransform, object)
def multi_latent_predict(XS, X, Y, likelihood, prior, diagonal):
    Ns = XS.shape[0]
    N = X.shape[0]
    P = Y.shape[1]

    K_xs = prior.vec_var(XS)[:, 0]
    K_xx = prior.full_covar(X, X)
    K_xs_x = prior.full_covar(XS, X)
    lik_var = get_diagonal_gaussian_likelihood_variances(Y, likelihood)
    mean_x = prior.vec_mean(X)
    mean_xs = prior.vec_mean(XS)

    Y_vec = Y.reshape(Y.shape[0]*Y.shape[1], 1, order='F')

    #TODO: implement masking

    mask = get_mask(Y_vec)
    Y_vec = np.nan_to_num(Y_vec,  nan=0.0)
    mask_xs_x = np.tile(mask, [K_xs_x.shape[0], 1]) 
    K_xs_x = np.multiply(K_xs_x, mask_xs_x)
    K_xx = mask_to_identity(K_xx, mask)
    mean_x =  mask_vector(mean_x, mask)

    mu, var = full_gaussian_prediction_diagonal(
        Y_vec,
        K_xs,
        K_xs_x,
        K_xx,
        mean_x,
        mean_xs,
        lik_var
    )   

    mu = mu.reshape([P, Ns])
    var = var.reshape([P, Ns])

    return mu, var
