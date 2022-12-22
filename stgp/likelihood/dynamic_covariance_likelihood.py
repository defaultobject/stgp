import objax
import jax.numpy as np
from . import Likelihood
from ..computation.gaussian import log_gaussian_scalar

class DynamicCovarianceLikelihood(Likelihood):
    pass

class DynamicCovarianceGaussian(DynamicCovarianceLikelihood):
    def log_likelihood_scalar(self, y, f):
        breakpoint()
