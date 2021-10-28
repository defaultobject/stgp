from ..kernels import Kernel, RBF
from ..likelihood import Gaussian
from ..dispatch import dispatch, evoke
from .gaussian import log_gaussian
from ..batching import loop_or_batch
from ..transforms import Independent, LinearTransform, LMC
from .model_ops import get_linear_multi_task_model_covariance
from .. import utils

import jax
import jax.numpy as np
from jax import jit
import chex
from typing import List
import objax
from objax import ModuleList
from typing import List

@dispatch(object, object, Gaussian, Kernel, object)
def log_marginal_likelihood(
    X: np.ndarray, Y: np.ndarray, likelihood: Gaussian, kernel: Kernel, mask: np.ndarray
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

    k_xx = kernel.K(X, X)
    lik_noise = likelihood.variance

    k = k_xx + lik_noise * np.eye(N)

    if False and (mask is not None):
        Y = np.nan_to_num(Y, nan=0.0)

        mask = np.tile(mask, [mask.shape[0], 1]) 

        k = k-np.eye(N)
        k = np.multiply(k, mask)
        k = np.multiply(k, mask.T)
        k = k+np.eye(N)

        return log_gaussian(Y, np.zeros_like(Y), k) - np.sum(1-mask)*(1/np.sqrt(2*np.pi))
        #return log_gaussian(Y, np.zeros_like(Y), k)

    return log_gaussian(Y, np.zeros_like(Y), k)


@dispatch(object, object, object, Independent, object)
def multi_latent_log_marginal_likelihood(
    X: np.ndarray, Y: np.ndarray, likelihood: list, prior: Independent, mask
):
    """ Independent Latent functions. Each marginal liklihood is computed separately and summed """

    # Assume that are likelihoods are the same such that they can be batched over

    lik = likelihood[0]

    num_latents = prior.num_latents
    num_outputs = prior.num_outputs

    # Extract kernels
    kernels = prior.get_kernels()

    lml_fn = evoke('log_marginal_likelihood')

    mask = mask.astype(float)

    def _lml(lml_fn, X, Y, lik,  kernel, mask):
        N = X.shape[0]
        Y = Y[:, None]

        return lml_fn(X, Y, lik, kernel, mask) 

    lml_arr = loop_or_batch(
        _lml,
        [ lml_fn, X, Y, likelihood, kernels, mask ],
        [ None, None, 1, 0, 0, 1],
        num_latents,
        num_returned_arguments=1
    )

    lml =  np.sum(lml_arr)

    return lml

@dispatch(object, object, object, LinearTransform, object)
def multi_latent_log_marginal_likelihood(
    X: np.ndarray, Y: np.ndarray, likelihood: List[Gaussian], prior: LinearTransform, mask=None
) -> np.ndarray:
    """
    The marginal likelihood is:
        p(Y)  = N(Y | 0, (W \kron I) K (W \kron I)^T + diag(eps_p))
    """

    assert all([type(lik) == Gaussian for lik in likelihood])

    N = Y.shape[0]

    sigma = get_linear_multi_task_model_covariance(X, Y, likelihood, prior)

    Y_vec = Y.reshape(Y.shape[0]*Y.shape[1], 1, order='F')


    if mask is not None:
        mask = mask.reshape(Y.shape[0]*Y.shape[1], order='F')
        Y_vec = Y_vec[mask]
        sigma = sigma[mask, ...]
        sigma = sigma[..., mask]

    return log_gaussian(Y_vec, np.zeros_like(Y_vec), sigma)

