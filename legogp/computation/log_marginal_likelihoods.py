from ..kernels import Kernel, RBF
from ..likelihood import Gaussian, GaussianParameterised
from ..dispatch import dispatch, evoke
from .gaussian import log_gaussian
from ..batching import loop_or_batch
from ..transforms import Independent, LinearTransform, LMC
from .model_ops import get_diagonal_gaussian_likelihood_variances
from ..utils import utils

from ..utils.nan_utils import mask_to_identity, get_mask, mask_vector

import jax
import jax.numpy as np
from jax import jit
import chex
from typing import List
import objax
from objax import ModuleList
from typing import List

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
    chex.assert_equal(Y.shape[1], 1)
    chex.assert_equal(Y.shape[0], X.shape[0])

    N = X.shape[0]

    lik_noise = likelihood.variance

    k = K + lik_noise * np.eye(N)

    mask = get_mask(Y)
    Y = np.nan_to_num(Y, nan=0.0)
    k = mask_to_identity(k, mask)
    mean = mask_vector(mean, mask)

    return log_gaussian(Y, mean, k) - np.sum(1-mask)*(1/np.sqrt(2*np.pi))


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


@dispatch(object, object, object, Independent)
def multi_latent_log_marginal_likelihood(
    X: np.ndarray, Y: np.ndarray, likelihood: list, prior: Independent
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

    Y = Y[..., None]

    lml_arr = loop_or_batch(
        lambda lml_fn, X, Y, lik, k, mean: lml_fn(X, Y, lik, k, mean[:, None]),
        [ lml_fn, X, Y, likelihood, k_xx_arr, mean_arr],
        [ None, None, 1, 0, 0, 0],
        num_latents,
        num_returned_args=1
    )

    lml =  np.sum(lml_arr)

    return lml

@dispatch(object, object, object, LinearTransform)
def multi_latent_log_marginal_likelihood(
    X: np.ndarray, Y: np.ndarray, likelihood: List[Gaussian], prior: LinearTransform
) -> np.ndarray:
    """
    The marginal likelihood is:
        p(Y)  = N(Y | 0, (W \kron I) K (W \kron I)^T + diag(eps_p))
    """

    assert all([type(lik) == Gaussian for lik in likelihood])

    N = Y.shape[0]

    Y_vec = Y.reshape(Y.shape[0]*Y.shape[1], 1, order='F')

    mean = prior.vec_mean(X)
    K_xx = prior.full_covar(X, X)
    lik_xx = get_diagonal_gaussian_likelihood_variances(Y, likelihood)

    sigma = K_xx + lik_xx

    #TODO: implement masking

    return log_gaussian(Y_vec, mean, sigma)

