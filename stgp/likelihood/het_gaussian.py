"""(Probably) Hetereogenous Gaussian likelihood."""
import objax
import jax.numpy as np
from . import DiagonalLikelihood
from ..computation.gaussian import log_gaussian_scalar


class HetGaussian(DiagonalLikelihood):
    def __init__(self):
        # hack for now, when using a mean-field it assumed that there is one likelihood
        #  per output
        self.likelihood_arr = [self]

    def log_likelihood_scalar(self, y, f):
        return log_gaussian_scalar(y, f[0], np.exp(f[1]))

    def conditional_var(self, f):
        return np.exp(f[1])

    def conditional_mean(self, f):
        return f[0]



