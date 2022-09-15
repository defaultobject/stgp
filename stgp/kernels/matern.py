import objax
import chex
import jax
import jax.numpy as np

from . import StationaryKernel, StationaryVarianceKernel, MarkovKernel


class ScaledMatern32(StationaryVarianceKernel, MarkovKernel):
    def to_ss(self, X_spatial=None):
        """ Return state space representation """
        chex.assert_equal(self.input_dim, 1)

        lengthscale = self.lengthscales[0]
        variance = self.variance

        # temporal so input dim in 1
        v = 3.0 / 2.0
        D = int(v + 0.5)

        lam = (3.0 ** 0.5) / self.lengthscales[0]
        F = np.array([[0.0, 1.0], [-(lam ** 2), -2 * lam]])

        L = np.array([[0.0], [1.0]])

        # measurement model matrix
        H = np.array([[1.0, 0.0]])

        Qc = np.array([
            [12.0 * 3.0 ** 0.5 / lengthscale ** 3.0 * self.variance]
        ])

        Pinf = np.array(
            [
                [1.0, 0.0],
                [0.0, 3.0 * self.variance / lengthscale ** 2.0],
            ]
        )

        return F, L, Qc, H, Pinf

    def state_size(self):
        return 2

    def expm(self, dt, X_spatial=None):
        """closed form matrix exponential A = expm(F * dt)"""
        chex.assert_equal(self.input_dim, 1)

        lam = np.sqrt(3.0) / self.lengthscales[0]
        A = np.exp(-dt * lam) * (dt * np.array([[lam, 1.0], [-lam**2.0, -lam]]) + np.eye(2))
        return A

    def _K_scaler_with_var(self, x1, x2, lengthscale, variance):
        """
        K(X1, X2) = σ (1 + √3 (X1-X2)/l) exp{-√3 (X1-X2)/l}
        """

        r  = np.abs(x1-x2) / lengthscale
        sqrt3 = np.sqrt(3.0)

        return  variance * (1.0 + sqrt3 * r) * np.exp(-sqrt3 * r)


class Matern32(StationaryKernel, MarkovKernel):
    def __init__(self, *args, **kwargs):
        super(Matern32, self).__init__(*args, **kwargs, name='Matern32')

    def to_ss(self, X_spatial=None):
        """ Return state space representation """
        chex.assert_equal(self.input_dim, 1)

        lengthscale = self.lengthscales[0]

        # temporal so input dim in 1
        v = 3.0 / 2.0
        D = int(v + 0.5)

        lam = (3.0 ** 0.5) / self.lengthscales[0]
        F = np.array([[0.0, 1.0], [-(lam ** 2), -2 * lam]])

        L = np.array([[0.0], [1.0]])

        # measurement model matrix
        H = np.array([[1.0, 0.0]])

        Qc = np.array([
            [12.0 * 3.0 ** 0.5 / lengthscale ** 3.0]
        ])

        Pinf = np.array(
            [
                [1.0, 0.0],
                [0.0, 3.0 * 1.0 / lengthscale ** 2.0],
            ]
        )

        return F, L, Qc, H, Pinf

    def state_size(self):
        return 2

    def expm(self, dt, X_spatial=None):
        """closed form matrix exponential A = expm(F * dt)"""
        chex.assert_equal(self.input_dim, 1)

        lam = np.sqrt(3.0) / self.lengthscales[0]
        A = np.exp(-dt * lam) * (dt * np.array([[lam, 1.0], [-lam**2.0, -lam]]) + np.eye(2))
        return A

    def _K_scaler(self, x1, x2, lengthscale):
        """
        K(X1, X2) = σ (1 + √3 (X1-X2)/l) exp{-√3 (X1-X2)/l}
        """

        r  = np.abs(x1-x2) / lengthscale
        sqrt3 = np.sqrt(3.0)

        return  (1.0 + sqrt3 * r) * np.exp(-sqrt3 * r)

class Matern12(StationaryKernel, MarkovKernel):
    def __init__(self, *args, **kwargs):
        super(Matern12, self).__init__(*args, **kwargs, name='Matern12')

    def cf_to_ss_temporal(self):
        chex.assert_equal(self.input_dim, 1)
        raise NotImplementedError()

    def _K_scaler(self, x1, x2, lengthscale):
        """
        K(X1, X2) = σ²  exp{-|X1-X2|/l}
        """

        return np.exp(- np.abs(x1-x2) / lengthscale)
