"""Gaussian likelihood."""
import objax
import jax.numpy as np
from . import Likelihood
from ..computation.parameter_transforms import inv_positive_transform, positive_transform
from ..batching import batch


class Gaussian(Likelihood):
    """Gaussian likelihood."""

    def __init__(self, variance=None):

        if variance is None:
            variance = 1.0

        self.raw_variance = objax.TrainVar(inv_positive_transform(variance))

    @batch
    def variance(self, raw_getter) -> np.ndarray:
        return positive_transform(raw_getter())

    def batched(self):
        pass


