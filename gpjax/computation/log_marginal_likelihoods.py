from ..dispatch import dispatch

import jax.numpy as np


@dispatch(np.ndarray, np.ndarrary, likelihood.Gaussian, kernel.Kernel)
def log_marginal_likelihood(
    X: np.ndarray, Y: np.ndarray, likelihood: likelihood.Gaussian, kernel: kernel.Kernel
):
    """
    Log marginal likelihood of GP prior with Gaussian likelihood.

    Computes:
        ...
    """

    print("gaussian likelihood log marginal likelihood")
