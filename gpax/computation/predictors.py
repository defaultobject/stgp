from ..settings import jitter
from ..kernels import Kernel, RBF
from ..likelihood import Gaussian
from ..approximate_posteriors import GaussianApproximatePosterior
from ..dispatch import dispatch, evoke
from .gaussian import log_gaussian
from ..batching import loop_or_batch
from ..transforms import Independent, LinearTransform
from .. import utils
from .matrix_ops import cholesky, log_chol_matrix_det, add_jitter, cholesky_solve
from .model_ops import get_block_diag_gram_matrix, get_diagonal_gaussian_likelihood_variances, get_linear_multi_task_model_covariance, get_linear_multi_task_prior_covariance, get_linear_multi_task_prior_diag_covariance

import jax
from jax import jit
import jax.numpy as np
import chex
from typing import List
from objax import ModuleList

@jit
def gaussian_prediction(Y, K_xs, K_xs_x, K_xx, lik_var):
    Ns = K_xs.shape[0]
    N = Y.shape[0]

    k = K_xx + np.eye(N)*lik_var

    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol, K_xs_x.T, lower=True)

    mu = K_xs_x @ cholesky_solve(k_chol, Y)
    sig = K_xs - A1.T @ A1

    mu = np.reshape(mu, [Ns, 1])
    sig = np.reshape(sig, [Ns, Ns])

    return mu, sig

@jit 
def full_gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, lik_xx):
    k = K_xx + lik_xx

    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol, K_xs_x.T, lower=True)

    mu = K_xs_x @ cholesky_solve(k_chol, Y)
    sig = K_xs - np.sum(np.square(A1), axis=0)
    sig = sig[:, None]

    chex.assert_equal(mu.shape, sig.shape)

    return mu, sig

@jit
def gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, lik_var):
    N = Y.shape
    lik_xx = np.eye(N)*lik_var

    return full_gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, lik_xx)
    

@dispatch(object, object, object, Gaussian, object)
def predict(XS, X, Y, likelihood, kernel):

    Ns = XS.shape[0]
    N = X.shape[0]

    K_xs = kernel.K(XS, XS)
    K_xs_x = kernel.K(XS, X)
    K_xx = kernel.K(X, X)

    chex.assert_equal(K_xx.shape, (X.shape[0], X.shape[0]))
    chex.assert_equal(K_xs_x.shape, (XS.shape[0], X.shape[0]))
    chex.assert_equal(K_xs.shape, (XS.shape[0], XS.shape[0]))

    return gaussian_prediction(Y, K_xs, K_xs_x, K_xx, likelihood.variance)

@dispatch(object, object, object, Gaussian, object)
def predict_diagonal(XS, X, Y, likelihood, kernel):

    Ns = XS.shape[0]
    N = X.shape[0]

    K_xs = kernel.K_diag(XS)
    K_xs_x = kernel.K(XS, X)
    K_xx = kernel.K(X, X)

    return  gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, likelihood.variance)



@dispatch(object, object, GaussianApproximatePosterior, Gaussian, object, object)
def predict_diagonal(XS, X,  approximate_posterior, likelihood, kernel, sparsity):
    m, S_diag = approximate_posterior.predictive_marginal(XS, X, kernel, sparsity, diagonal=True)
    return m, S_diag + likelihood.variance

@dispatch(object, object, GaussianApproximatePosterior, Gaussian, object, object)
def predict_full(XS, X,  approximate_posterior, likelihood, kernel, sparsity):
    m, S = approximate_posterior.predictive_marginal(XS, X, kernel, sparsity, diagonal=False)
    return m, S + np.eye(XS.shape[0])*likelihood.variance

@dispatch(object, object, object, object, Independent, object)
def multi_latent_predict(XS, X, Y, likelihood, prior, diagonal):
    """ Independent Latent functions. Each predictions is computed separately"""

    # Assume that are likelihoods are the same such that they can be batched over
    lik = likelihood[0]

    num_latents = prior.num_latents
    num_outputs = prior.num_outputs

    if diagonal:
        pred_fn = evoke('predict_diagonal')
    else:
        pred_fn = evoke('predict_diagonal')

    # Extract kernels
    kernels = prior.get_kernels()

    def _predict(pred_fn, XS, X, Y, lik,  kernel):
        Y = Y[:, None]
        return pred_fn(XS, X, Y, lik, kernel)

    mu_arr, var_arr = loop_or_batch(
        _predict,
        [pred_fn, XS, X, Y, likelihood, kernels],
        [None, None, None, 1, 0, 0],
        num_latents,
        num_returned_arguments=2
    )

    return mu_arr, var_arr

@dispatch(object, object, object, object, LinearTransform, object)
def multi_latent_predict(XS, X, Y, likelihood, prior, diagonal):
    Ns = XS.shape[0]
    N = X.shape[0]
    P = Y.shape[1]

    kernels = prior.get_kernels()

    #sigma_x = get_linear_multi_task_model_covariance(X, Y, likelihood, prior)

    #XXS = np.vstack([X, XS])


    #sigma_xs = get_linear_multi_task_model_covariance(XXS, Y, likelihood, prior)
    #breakpoint()

    K_xs = get_linear_multi_task_prior_diag_covariance(XS, prior)
    K_xx = get_linear_multi_task_prior_covariance(X, X, prior)
    K_xs_x = get_linear_multi_task_prior_covariance(XS, X, prior)
    lik_var = get_diagonal_gaussian_likelihood_variances(Y, likelihood)

    Y_vec = Y.reshape(Y.shape[0]*Y.shape[1], 1, order='F')

    mu, var = full_gaussian_prediction_diagonal(
        Y_vec,
        K_xs,
        K_xs_x,
        K_xx,
        lik_var
    )   

    mu = mu.reshape([P, Ns])
    var = var.reshape([P, Ns])

    return mu, var
