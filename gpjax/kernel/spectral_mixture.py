import objax
import chex
import jax
import jax.numpy as np

from . import StationaryKernel
from typing import List, Optional, Union
from ..computation.parameter_transforms import inv_positive_transform, positive_transform
from ..batching import batch


class SM_Component(StationaryKernel):
    """
        We re-use lengthscale to mean \mu
                   variance to be v
    """

    def __init__(
        self,
        lengthscales: Optional[np.ndarray] = None,
        variance: Optional[np.ndarray] = None,
        input_dim: Optional[int] = 1,
        active_dims: Optional[np.ndarray] = None,
    ) -> None:

        super(StationaryKernel, self).__init__(input_dim, active_dims)

        # input admin
        if lengthscales is None:
            lengthscales = np.array([1.0] * input_dim)
        else:
            lengthscales = ensure_array(lengthscales)

        if variance is None:
            variance = 1.0
        else:
            ensure_float(variance)

        chex.assert_shape(lengthscales, [input_dim])
        chex.assert_rank(variance, 0)  # scalar

        # register lengthscales and variances
        self.raw_lengthscales = objax.TrainVar(lengthscales)
        self.raw_variance = objax.TrainVar(inv_positive_transform(variance))


    @batch
    def lengthscales(self, raw_getter) -> np.ndarray:
        """ \mu is not constrained to be positive in the SM kernel. """
        return raw_getter()

    def _K_scaler(self, x1, x2, variance, lengthscale):
        #ensure scalar inputs
        chex.assert_rank(x1, 0)
        chex.assert_rank(x2, 0)
        chex.assert_rank(variance, 0)
        chex.assert_rank(lengthscale, 0)

        mu = lengthscale
        v = variance

        tau = x1-x2
        tau2 = tau**2

        return np.exp(-2*np.pi*tau2*v)*np.cos(2*np.pi*tau*mu)
