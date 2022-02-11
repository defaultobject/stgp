from ..kernels import Kernel, RBF
from ..likelihood import Gaussian, GaussianParameterised, ProductLikelihood
from ..dispatch import dispatch, evoke
from .gaussian import log_gaussian, log_gaussian_with_nans
from ..transforms import Independent, LinearTransform
from .model_ops import get_diagonal_gaussian_likelihood_variances
from .matrix_ops import vec_columns
from ..models import BatchGP
from ..utils import utils
from ..utils.utils import get_batch_type


from ..utils.nan_utils import mask_to_identity, get_mask, mask_vector
from ..utils.utils import can_batch

import jax
import jax.numpy as np
from jax import jit
import chex
from typing import List
import objax
from objax import ModuleList
from typing import List
from batchjax import batch_or_loop, BatchType

@dispatch(object, object, Gaussian, object, object)
def log_marginal_likelihood(
        X: np.ndarray, Y: np.ndarray, likelihood: Gaussian, K: np.ndarray, mean: np.ndarray
):
    """
    Log marginal likelihood of GP prior with Gaussian likelihood.

    Computes:
        log N(Y | 0, K(X, X) + lik.variance*I)

    """
    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)

    N = X.shape[0]

    chex.assert_shape(Y, [N, 1])
    chex.assert_shape(mean, [N, 1])
    chex.assert_shape(K, [N, N])

    lik_noise = likelihood.variance

    k = K + lik_noise * np.eye(N)

    return log_gaussian_with_nans(Y, mean, k) 


@dispatch(object, object, GaussianParameterised, Kernel, object)
def log_marginal_likelihood(
    X: np.ndarray, Y: np.ndarray, likelihood: GaussianParameterised, kernel: Kernel, mask: np.ndarray
):

    N = X.shape[0]

    k_xx = kernel.K(X, X)
    k = k_xx + likelihood.variance(X)

    if (mask is not None):
        Y = np.nan_to_num(Y, nan=0.0)
        k = mask_to_identity(k, mask)

        return log_gaussian(Y, np.zeros_like(Y), k) - np.sum(1-mask)*(1/np.sqrt(2*np.pi))

    return log_gaussian(Y, np.zeros_like(Y), k)

@dispatch(BatchGP, ProductLikelihood, LinearTransform)
def log_marginal_likelihood(
        X: np.ndarray, Y: np.ndarray, gp: 'Posterior', likelihood: ProductLikelihood, prior: LinearTransform
):
    """ Independent Latent functions. Each marginal liklihood is computed separately and summed """

    likelihood_arr = likelihood.likelihood_arr
    # Assume that are likelihoods are the same such that they can be batched over
    assert all([type(lik) == Gaussian for lik in likelihood_arr])

    N = Y.shape[0]

    Y_vec = vec_columns(Y)

    mean = prior.vec_mean(X)
    K_xx = prior.full_covar(X, X)
    lik_xx = get_diagonal_gaussian_likelihood_variances(Y, likelihood_arr)

    sigma = K_xx + lik_xx

    return log_gaussian_with_nans(Y_vec, mean, sigma) 


@dispatch(BatchGP, ProductLikelihood, Independent)
def log_marginal_likelihood(
        X: np.ndarray, Y: np.ndarray, gp: 'Posterior', likelihood: ProductLikelihood, prior: Independent
):
    """ Independent Latent functions. Each marginal liklihood is computed separately and summed """

    # Assume that are likelihoods are the same such that they can be batched over

    num_latents = prior.num_latents
    num_outputs = prior.num_outputs

    # precompute prior covariance
    k_xx_arr = prior.covar(X, X)
    mean_arr = prior.mean(X) 

    # get correct marginal likelihood from dispatch
    lml_fn = evoke('log_marginal_likelihood')

    # Ensure batched Y has rank 2
    Y = Y[..., None]

    likelihood_arr = likelihood.likelihood_arr

    # Compute lml for each likelihood and prior
    lml_arr = batch_or_loop(
        lambda lml_fn, X, Y, lik, k, mean: lml_fn(X, Y, lik, k, mean),
        [ lml_fn, X, Y, likelihood_arr, k_xx_arr, mean_arr],
        [ None, None, 1, 0, 0, 0],
        dim = num_latents,
        out_dim = 1,
        batch_type = get_batch_type(likelihood_arr)
    )

    lml =  np.sum(lml_arr)

    return lml
