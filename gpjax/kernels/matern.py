import objax
import jax
import jax.numpy as np

from . import StationaryKernel, MarkovKernel

class Matern32(StationaryKernel, MarkovKernel):
    def K(self, X1, X2, active_dims=None):
        """
                K(X1, X2) = σ² (1 + √3 (X1-X2)/l) exp{-√3 (X1-X2)/l}
        """

        diff = X1-X2.T
        r2 = np.square(diff / self.lengthscale[i])
        r = np.sqrt(np.clip(r2, 1e-36))

        sqrt3 = np.sqrt(3.0)

        return self.variance[i] * (1.0 + sqrt3 * r) * np.exp(-sqrt3 * r)

    def K_diag(self, X1):
        return np.prod(self.variances)*np.ones(X1.shape[0])
