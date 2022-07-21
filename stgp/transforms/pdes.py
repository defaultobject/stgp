import jax.numpy as np

# Import Types
from . import Transform, LinearTransform


class DifferentialOperatorJoint(LinearTransform):
    """
    tbd.
    """
    def __init__(
        self,
        base_latent,
        derivative_kernel
    ):
        if base_latent is None:
            raise RuntimeError('Latent gp must be passed')

        self._parent = base_latent
        self.derivative_kernel = derivative_kernel
        self._output_dim = self.derivative_kernel.output_dim

    def mean(self, X1):
        return np.zeros([X1.shape[0] * self.output_dim, 1])

    def mean_blocks(self, X1):
        return np.zeros([self.output_dim, X1.shape[0], 1])

    def covar(self, X1, X2):
        return self.derivative_kernel.K(X1, X2)

class PDE(Transform):
    pass


class HeatEquation2D(PDE, LinearTransform):
    def __init__(self, latent):
        """
            latent must be a DifferentialOperatorJoint with a 2D differential kernel.
        """
        self._parent = latent

    def mean(self, X):
        raise NotImplementedError()

    def full_var(self, X):
        raise NotImplementedError()

