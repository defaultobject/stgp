from ..kernels import Kernel, RBF
from ..likelihood import Gaussian
from ..dispatch import dispatch, evoke
from .gaussian import log_gaussian
from ..batching import loop_or_batch
from ..transforms import Independent
from .. import utils

import jax
import jax.numpy as np
from jax import jit
import chex
from typing import List
import objax
from objax import ModuleList

@dispatch(object, object, Gaussian, Kernel)
def log_marginal_likelihood(
    X: np.ndarray, Y: np.ndarray, likelihood: Gaussian, kernel: Kernel
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

    return log_gaussian(Y, np.zeros_like(Y), k)


@dispatch(object, object, object, Independent)
def multi_latent_log_marginal_likelihood(
    X: np.ndarray, Y: np.ndarray, likelihood: list, prior: Independent
):
    """ Independent Latent functions. Each marginal liklihood is computed separately and summed """

    # Assume that are likelihoods are the same such that they can be batched over

    lik = likelihood[0]

    num_latents = prior.num_latents
    num_outputs = prior.num_outputs

    # Extract kernels
    kernels = prior.get_kernels()

    lml_fn = evoke('log_marginal_likelihood')

    def _lml(lml_fn, X, Y, lik,  kernel):
        N = X.shape[0]
        Y = Y[:, None]

        return lml_fn(X, Y, lik, kernel)

    lml_arr = loop_or_batch(
        _lml,
        [ lml_fn, X, Y, likelihood, kernels ],
        [ None, None, 1, 0, 0 ],
        num_latents,
        num_returned_arguments=1
    )

    lml =  np.sum(lml_arr)

    return lml
