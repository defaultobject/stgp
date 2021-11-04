"""Gaussian likelihood."""
import objax
import jax.numpy as np
from . import Likelihood
from ..computation.parameter_transforms import inv_positive_transform, positive_transform
from ..batching import batch

from ..computation.gaussian import log_gaussian_scalar



class Gaussian(Likelihood):
    """Gaussian likelihood."""

    def __init__(self, variance=None):

        if variance is None:
            variance = 1.0

        #self.raw_variance = objax.StateVar(inv_positive_transform(variance))
        self.raw_variance = objax.TrainVar(inv_positive_transform(variance))

    @batch
    def variance(self, raw_getter) -> np.ndarray:
        return positive_transform(raw_getter())

    def log_likelihood_scalar(self, y, f):
        var = np.array([self.variance])
        ll = log_gaussian_scalar(y, f, var)
        return ll

    def conditional_var(self, f):
        return self.variance

    def conditional_mean(self, f):
        return f

class GaussianParameterised(Likelihood):
    """Gaussian likelihood."""

    def __init__(self, kernel):
        self.kernel = kernel

    def variance(self, X) -> np.ndarray:
        return self.kernel.K(X, X)
