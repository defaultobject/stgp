"""Gaussian likelihood."""
import objax
import jax.numpy as np
from . import Likelihood, DiagonalLikelihood

from .. import Parameter

from ..computation.parameter_transforms import inv_positive_transform, positive_transform
from ..computation.gaussian import log_gaussian_scalar



class Gaussian(DiagonalLikelihood):
    """Gaussian likelihood."""

    def __init__(self, variance=None):

        if variance is None:
            variance = 1.0

        self.variance_param = Parameter(variance, constraint='positive')

    @property
    def variance(self) -> np.ndarray:
        return self.variance_param.value

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
