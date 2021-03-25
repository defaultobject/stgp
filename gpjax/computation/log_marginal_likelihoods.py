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
