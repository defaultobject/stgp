"""Bernoulli likelihood."""
import objax
import jax.numpy as np
from . import DiagonalLikelihood
from ..computation.parameter_transforms import inv_positive_transform, positive_transform, inv_probit
from ..computation.general import log_bernoulli


class Bernoulli(DiagonalLikelihood):
    """Bernoulli likelihood."""

    def __init__(self, link_fn = inv_probit):
        self.link_fn = link_fn

    def log_likelihood_scalar(self, y, f):
        ll = log_bernoulli(y, self.link_fn(f))
        return ll

    def conditional_var(self, f):
        return self.conditional_mean(f)

    def conditional_mean(self, f):
        sig_f = self.link_fn(f)
        return sig_f * (1 - sig_f)



