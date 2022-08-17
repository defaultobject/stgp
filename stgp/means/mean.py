import objax
import chex
import jax
import jax.numpy as np
from ..computation.matrix_ops import hessian
from jax import jacfwd, jacrev, grad

class Mean(objax.Module):
    pass

class ZeroMean(Mean):
    def __init__(self, parent_model):
        self.output_dim = 1

    def mean_blocks(self, X):
        N = X.shape[0]
        return np.zeros([self.output_dim, N, 1])

    def mean(self, X):
        N = X.shape[0]
        return np.zeros([self.output_dim * N, 1])

class DiffOpMean(Mean):
    pass

class SecondOrderDerivativeMean_1D(DiffOpMean):
    """
    Given mu(x) computes [mu(x), dmu(x)/dx, d^2mu(x)/dx^2]
    """
    def __init__(self, parent_model = None):
        self.output_dim = 3
        self.parent_model = parent_model

    def mean_blocks_from_fn(self, X, mean_fn):
        # assumes output dim is
        fn = lambda XS: mean_fn(XS)[:, 0]

        mu_x = fn(X)
        dmu_dx = jax.vmap(jacfwd(fn))(X[:, None, :])
        d2mu_dx2 = jax.vmap(hessian(fn, 0))(X[:, None, :])

        dmu_dx = np.squeeze(dmu_dx)
        d2mu_dx2 = np.squeeze(d2mu_dx2)

        # return rank 3 matrix
        return np.array([
            mu_x, 
            dmu_dx, 
            d2mu_dx2
        ])[..., None]

    def mean_from_fn(self, X, mean_fn):
        return np.vstack(self.mean_blocks_from_fn(X, mean_fn))

