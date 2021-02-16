import objax
import chex
import jax
import jax.numpy as np

from . import StationaryKernel, MarkovKernel

class Matern32(StationaryKernel, MarkovKernel):

    def cf_to_ss_temporal(self):
        chex.assert_equal(self.input_dim, 1)
        raise NotImplementedError()

        #temporal so input dim in 1
        v = 3.0/2.0
        D = int(v+0.5)


        lam = (3.0 ** 0.5) / self.lengthscales[0]
        F = np.array([[0.0,       1.0],
                      [-lam ** 2, -2 * lam]])

        L = np.array([
            [0.0],
            [1.0]
        ])

        #measurement model matrix
        H = np.array([[1.0, 0.0]])

        Qc = np.array(12.0 * 3.0 ** 0.5 / self.lengthscales[0] ** 3.0 * self.variance)

        Pinf = np.array([[self.variance, 0.0],
                         [0.0, 3.0 * self.variance / self.lengthscale[0] ** 2.0]])
        
        return F, L, Qc, H, Pinf


    def _K(self, X1, X2):
        """
                K(X1, X2) = σ² (1 + √3 (X1-X2)/l) exp{-√3 (X1-X2)/l}
        """

        diff = X1-X2.T
        r2 = np.square(diff / self.lengthscales)
        r = np.sqrt(np.clip(r2, 1e-36))

        sqrt3 = np.sqrt(3.0)

        return self.variance * (1.0 + sqrt3 * r) * np.exp(-sqrt3 * r)


