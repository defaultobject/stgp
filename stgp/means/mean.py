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
        N = X.shape[0]

        # assumes output dim is
        fn = lambda XS: mean_fn(XS)[:, 0]

        mu_x = fn(X)
        dmu_dx = jax.vmap(jacfwd(fn))(X[:, None, :])
        d2mu_dx2 = jax.vmap(hessian(fn, 0))(X[:, None, :])

        mu_x = np.squeeze(mu_x)
        dmu_dx = np.squeeze(dmu_dx)
        d2mu_dx2 = np.squeeze(d2mu_dx2)

        # return rank 3 matrix
        d_mu =  np.array([
            mu_x, 
            dmu_dx, 
            d2mu_dx2
        ])[..., None]

        d_mu = np.transpose(d_mu, [1, 0, 2])
        chex.assert_shape(d_mu, [N, self.output_dim, 1])

        return d_mu

    def mean_from_fn(self, X, mean_fn):
        return np.vstack(self.mean_blocks_from_fn(X, mean_fn))


class SecondOrderDerivativeMean_2D(DiffOpMean):
    """
    Given mu(t, x) computes [
        mu(t, x), 
        d mu(x)/dt, 
        d^2 mu(x)/dt^2,
        d mu(x)/dx, 
        d^2 mu(x)/dx^2
    ]
    """
    def __init__(
            self, 
            parent_model = None
        ):

        self.output_dim = 5
        self.parent_model = parent_model

    def mean_blocks_from_fn(self, X, mean_fn):
        N = X.shape[0]

        # assumes output dim is
        fn = lambda xs: mean_fn(xs[None, :])

        mu_x = jax.vmap(fn)(X)
        dmu = jax.vmap(jacfwd(fn, argnums=0))(X)
        d2mu = jax.vmap(hessian(fn, 0))(X)

        mu_x = np.squeeze(mu_x)
        dmu_dt = np.squeeze(dmu[..., 0])
        d2mu_dt2 = np.squeeze(d2mu[..., 0, 0])
        dmu_dx = np.squeeze(dmu[..., 1])
        d2mu_dx2 = np.squeeze(d2mu[..., 1, 1])

        # return rank 3 matrix
        d_mu =  np.array([
            mu_x, 
            dmu_dt, 
            d2mu_dt2,
            dmu_dx, 
            d2mu_dx2
        ])[..., None]

        d_mu = np.transpose(d_mu, [1, 0, 2])
        chex.assert_shape(d_mu, [N, 5, 1])

        return d_mu

    def mean_from_fn(self, X, mean_fn):
        return np.vstack(self.mean_blocks_from_fn(X, mean_fn))

class SecondOrderSpaceFirstOrderTimeDerivativeMean_2D(SecondOrderDerivativeMean_2D):
    """
    Given mu(t, x) computes [
        mu(t, x), 
        d mu(x)/dt, 
        d^2 mu(x)/dx^2
    ]
    """
    def __init__(
            self, 
            parent_model = None
        ):

        self.output_dim = 3
        self.parent_model = parent_model

    def mean_blocks_from_fn(self, X, mean_fn):
        blocks = super(SecondOrderSpaceFirstOrderTimeDerivativeMean_2D, self).mean_blocks_from_fn(X, mean_fn)
        return blocks[:, [0,1,4], :]


    def mean_from_fn(self, X, mean_fn):
        return np.vstack(self.mean_blocks_from_fn(X, mean_fn))
