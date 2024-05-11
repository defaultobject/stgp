import jax
import jax.numpy as np
import chex
from . import NonStationaryKernel


class ArcCosine(NonStationaryKernel):
    def __init__(self, order=0, active_dims=None):
        self.order = order
        self.active_dims = None

    def get_theta(self, x, y):
        chex.assert_rank(x, 2)
        chex.assert_rank(y, 2)
        normed_dot = (x.T @ y)/(np.linalg.norm(x, ord=2)*np.linalg.norm(y, ord=2))
        return np.squeeze(np.arccos(np.clip(normed_dot, -1, 1)))

    def get_J(self, theta):
        if self.order == 0:
            return np.pi - theta
        elif self.order == 1:
            return np.sin(theta)+(np.pi-theta)*np.cos(theta)
        elif self.order == 2:
            return 3*np.sin(theta)*np.cos(theta)+(np.pi-theta)*(1+2*(np.cos(theta)**2))
        raise RuntimeError()

    def add_bias(self, x):
        return np.stack([x, np.ones_like(x)])[:, None]
    def K_diag(self, X):
        K = jax.vmap(lambda x: self._K_scaler(np.squeeze(x), np.squeeze(x)))(X)
        chex.assert_rank(K, 1)
        return K
        
    def _K_scaler(self, x, y):
        x = self.add_bias(x)
        y = self.add_bias(y)

        theta = self.get_theta(x, y)

        J = self.get_J(theta)
        x_norm = np.linalg.norm(x, ord=2)
        y_norm = np.linalg.norm(y, ord=2)

        K = (1/np.pi) * np.pow(x_norm, self.order)*np.pow(y_norm, self.order)*J
        return np.squeeze(K)
