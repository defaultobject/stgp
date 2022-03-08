"""Gaussian likelihood."""
import objax
import jax.numpy as np
from . import DiagonalLikelihood
from ..computation.parameter_transforms import inv_positive_transform, positive_transform
from ..computation.general import log_poisson


class Poisson(DiagonalLikelihood):
    """Poisson likelihood."""

    def __init__(self, binsize):
        self.binsize = binsize
        self.link_fn = np.exp

    def log_likelihood(self, y, f):
        ll = log_poisson(y, self.link_fn(f) * self.binsize)
        return ll

    def conditional_var(self, f):
        return self.conditional_mean(f)

    def conditional_mean(self, f):
        return self.link_fn(f) * self.binsize




