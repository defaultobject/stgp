from ..settings import jitter
from ..kernels import Kernel, RBF
from ..likelihood import Gaussian, GaussianParameterised, ProductLikelihood
from ..approximate_posteriors import GaussianApproximatePosterior, MeanFieldApproximatePosterior
from ..dispatch import dispatch, evoke
from .gaussian import log_gaussian
from ..batching import loop_or_batch
from ..transforms import Independent, LinearTransform
from .elbos import precompute_variational_primitives, precompute_diagonal_variational_primitives
from .marginals import gaussian_conditional_diagional, gaussian_conditional_covar

from ..utils import utils
from ..utils.utils import can_batch, get_batch_type
from ..utils.nan_utils import mask_to_identity, get_mask, mask_vector

from .matrix_ops import cholesky, log_chol_matrix_det, add_jitter, cholesky_solve, vec_columns
from .model_ops import get_block_diag_gram_matrix, get_diagonal_gaussian_likelihood_variances, get_linear_multi_task_model_covariance, get_linear_multi_task_prior_covariance, get_linear_multi_task_prior_diag_covariance

import jax
from jax import jit
import jax.numpy as np
import chex
from typing import List
from objax import ModuleList
from batchjax import batch_or_loop, BatchType

@jit
def gp_posterior_with_nans(Y, K_xx, mean_x, lik_var):
    N = Y.shape[0]
    mask = get_mask(Y)

    Y = np.nan_to_num(Y, nan=0.0)
    k = K_xx + np.eye(N)*lik_var

    k = mask_to_identity(k, mask)
    k_chol = cholesky(k)

    m = mean_x + K_xx @ cholesky_solve(k_chol, Y-mean_x) 
    S = np.eye(N)

    return m, S

@jit
def gaussian_predictive_mean(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
    Ns = K_xs.shape[0]
    N = Y.shape[0]

    k = add_jitter(K_xx, lik_var)
    k_chol = cholesky(k)

    mu = K_xs_x @ cholesky_solve(k_chol, Y-mean_x) + mean_xs
    mu = np.reshape(mu, [Ns, 1])

    return mu

@jit
def gaussian_predictive_covar(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
    Ns = K_xs.shape[0]
    N = Y.shape[0]

    k = add_jitter(K_xx, lik_var)
    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol, K_xs_x.T, lower=True)
    sig = K_xs - A1.T @ A1
    sig = np.reshape(sig, [Ns, Ns])

    return sig

@jit
def gaussian_prediction(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
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
def gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
    k = add_jitter(K_xx, lik_var)

    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol, K_xs_x.T, lower=True)

    mu = K_xs_x @ cholesky_solve(k_chol, Y-mean_x) + mean_xs
    sig = K_xs - np.sum(np.square(A1), axis=0)
    sig = sig[:, None]

    chex.assert_equal(mu.shape, sig.shape)

    return mu, sig


@dispatch('BatchGP', Gaussian)
def predict(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs):
    chex.assert_equal(K_xx.shape, (X.shape[0], X.shape[0]))
    chex.assert_equal(K_xs_x.shape, (XS.shape[0], X.shape[0]))
    chex.assert_equal(K_xs.shape, (XS.shape[0], XS.shape[0]))

    # TODO: nans?

    return gaussian_prediction(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, likelihood.variance)

@dispatch('BatchGP', Gaussian)
def predict_diagonal(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs):

    #TODO: masking
    if False:
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

    var_arr = batch_or_loop(
        pred_fn,
        [Y, K_xs, K_xs_x, K_x_xs, K_xx, likelihood],
        [1, 0, 0, 0, 0, 0, 0],
        dim=num_latents,
        out_dim=1,
        batch_type = BatchType.LOOP
    )

    chex.assert_shape(var_arr, [num_outputs, XS_1.shape[0], XS_2.shape[0]])

    return var_arr

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

    print('predict -- LinearTransform -- here')

    Y_vec = vec_columns(Y)

    m_post, S_post = gp_posterior_with_nans(Y_vec, K_xx, mean_x, lik_var)

    mu, var = gaussian_conditional_diagional(
        XS, 
        X,
        K_xx,
        K_xs_x,
        K_xs,
        m_post,
        S_post,
        mean_x,
        mean_xs
    )

    mu = mu.reshape([P, Ns])
    var = var.reshape([P, Ns])


    return mu, var


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

    # if all likelihooods are the same we only need the first object
    #   and then we can batch it
    # otherwises we need the whole array and we will loop through them all
    if can_batch(likelihood_arr):
        pred_fn = evoke(evoke_name, gp, likelihood_arr[0])
        pred_axes = None
    else:
        pred_fn = [evoke(evoke_name, gp, lik) for lik in likelihood_arr]
        pred_axes = 0

    fn = lambda pred_fn, *args: pred_fn(*args)

    # Ensure Y is rank 2 after batching
    Y = Y[..., None]

    # Compute prediction for each likelihood-prior pair
    mu_arr, var_arr = batch_or_loop(
        fn,
        [pred_fn, XS, X, Y, likelihood_arr, K_xs, K_xs_x, K_xx, mean_x, mean_xs],
        [pred_axes, None, None, 1, 0, 0, 0, 0, 0, 0],
        dim=num_latents,
        out_dim=2,
        batch_type = get_batch_type(likelihood_arr)
    )

    return mu_arr, var_arr
