from ..kernel import Kernel, RBF
from ..likelihood import Gaussian
from ..dispatch import dispatch
from .gaussian import log_gaussian
from ..batching import Batched
from .. import utils

import jax
import jax.numpy as np
import chex
from typing import List
from objax import ModuleList


@dispatch(object, object, Gaussian, Kernel)
def log_marginal_likelihood(
    X: np.ndarray, Y: np.ndarray, likelihood: Gaussian, kernel: Kernel
):
    """
    Log marginal likelihood of GP prior with Gaussian likelihood.

    Computes:
        ...

    """
    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_equal(Y.shape[1], 1)

    N = X.shape[0]

    k_xx = kernel.K(X, X)
    lik_noise = likelihood.variance

    k = k_xx + lik_noise * np.eye(N)

    return log_gaussian(Y, np.zeros_like(Y), k)

#batched log marginal likelihood
@dispatch(object, object, Gaussian, ModuleList, ModuleList)
def log_marginal_likelihood(
        X: np.ndarray, Y: np.ndarray, tmp_lik: Gaussian, likelihood: List[Gaussian], kernel: List[Kernel]
):
    """
    Log marginal likelihood of GP prior with Gaussian likelihood.

    Computes:
        ...

    """

    with Batched(likelihood) as likelihood, Batched(kernel) as kernel:

        def lml(X, Y, lik, lik_vars, kernel, kernel_vars):
            N = X.shape[0]
            Y = Y[:, None]

            lik.set_vars(lik_vars)
            kernel.set_vars(kernel_vars)

            lik_noise = lik.get_obj().variance
            K_xx = kernel.get_obj().K(X, X)

            k = K_xx + lik_noise * np.eye(N)

            return log_gaussian(Y, np.zeros_like(Y), k)

        lml = jax.vmap(lml, (None, 1, None, 0, None, 0))(X, Y, likelihood, likelihood.get_vars(), kernel, kernel.get_vars())


    return lml
