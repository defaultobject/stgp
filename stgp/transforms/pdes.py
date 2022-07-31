import jax
import jax.numpy as np
import chex

# Import Types
from . import Transform, LinearTransform, Joint
from ..computation.matrix_ops import get_block_diagonal


class DifferentialOperatorJoint(LinearTransform, Joint):
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

    def forward(self, f): return f

    def transform(self, mu, var):
        """ This is a base prior so no need to transform """
        chex.assert_rank([mu, var], [2, 2])
        return mu, var

    def get_sparsity_list(self):
        return [self.get_sparsity()]

    def get_Z(self):
        Z = self.parent.get_Z()
        Z_all = np.tile(Z, [self.output_dim, 1, 1])
        return Z_all

    def get_sparsity(self):
        return self.parent.sparsity

    def mean(self, X1):
        return np.zeros([X1.shape[0] * self.output_dim, 1])

    def mean_blocks(self, X1):
        return np.zeros([self.output_dim, X1.shape[0], 1])

    def b_mean(self, X1):
        # We only support zero mean so we simply return the mean
        return self.mean(X1[0])

    def b_mean_blocks(self, X1):
        return self.mean_blocks(X1[0])

    def covar(self, X1, X2):
        return self.derivative_kernel.K(X1, X2)

    def b_covar(self, X1, X2):
        """ WARNING: we assume that X1, X2 is actually repeated """

        return self.covar(X1[0], X2[0])

    def covar_blocks(self, X1, X2):
        K_full =  self.derivative_kernel.K(X1, X2)

        return get_block_diagonal(
            K_full,
            self.output_dim,
        )

    def b_covar_blocks(self, X1, X2):
        """ WARNING: we assume that X1 is actually repeated """

        return self.covar_blocks(X1[0], X2[0])

    def full_var(self, X):
        return self.covar(X, X)

    def var(self, X):
        fn = jax.vmap(lambda p,x: p.covar(x[None, :], x[None, :]), [None, 0])
        res = fn(self, X)
        diag_vec = np.diagonal(res, axis1=1, axis2=2)
        res =  np.hstack(diag_vec.T)[:, None]

        chex.assert_rank(res, 2)
        return res

    @property
    def base_prior(self):
        """ DifferentialOperatorJoint is only used to construct a base prior""" 
        return self

class PDE(Transform):
    pass

class Pendulum1D(PDE):
    def __init__(self, latent):
        """
        Latent must be a DifferentialOperatorJoint with a 1D differential kernel.

        Let x be the angle

        The  transform is:
            d^2 x/dt^2 + sin(x) = 0
        """
        self._parent = latent
        self._output_dim = 1
        self._input_dim = self.parent.output_dim

        self.W = np.eye(1)

    def forward(self, f):
        """ 
        f is of shape 5 corresponding to f, ft, ft2
        """
        t = f[0]
        dt2 = f[2]

        res = dt2 + np.sin(t)

        return np.array([res])


class HeatEquation2D(PDE, LinearTransform):
    def __init__(self, latent):
        """
        Latent must be a DifferentialOperatorJoint with a 2D differential kernel.

        The heat kernel transform is:
            df/dt - d^2f/dx^2 = 0
        """
        self._parent = latent
        self._output_dim = 1
        self._input_dim = self.parent.output_dim

        self.W = np.eye(1)

    def forward(self, f):
        """ 
        f is of shape 5 corresponding to f, ft, ft2, fx, fx2
        """
        return np.array([f[1] - f[4]])

    def _transform_mean(self, mu):

        dt = mu[1]
        dx2 = mu[4]

        return np.array([dt - dx2])

    def _transform_covar(self, var):

        N1 = int(var.shape[0]/self.input_dim)
        N2 = int(var.shape[1]/self.input_dim)

        left_idx = np.arange(N1)
        right_idx = np.arange(N2)

        Kt = var[left_idx+N1*1, :]
        Kt = Kt[:, right_idx+N2*1]

        Kx2 = var[left_idx+N1*4, :]
        Kx2 = Kx2[:, right_idx+N2*4]

        Ktx2 = var[left_idx+N1*1, :]
        Ktx2 = Ktx2[:, right_idx+N2*4]

        Kx2t = var[left_idx+N1*4, :]
        Kx2t = Kx2t[:, right_idx+N2*1]

        K =  Kt + Kx2 - Ktx2 - Kx2t

        #ensure correct rank
        chex.assert_shape(K, [N1 * self.output_dim, N2 * self.output_dim]) 

        return K

    def transform(self, mu, var):
        return self._transform_mean(mu), self._transform_covar(var)


    def mean(self, X):
        # parent is a DifferentialOperatorJoint whose outputs are:
        #   f, ft, ft2, fx, fx2 
        parent_mean = self.parent.mean_blocks(X)

        return self._transform_mean(parent_mean)

    def covar(self, X1, X2):
        parent_covar = self.parent.covar(X1, X2)

        return self._transform_covar(parent_covar)


