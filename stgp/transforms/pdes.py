import jax.numpy as np
import chex

# Import Types
from . import Transform, LinearTransform, Joint


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

    def get_Z(self):
        return self.parent.get_Z()

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

    @property
    def base_prior(self):
        """ DifferentialOperatorJoint is only used to construct a base prior""" 
        return self

class PDE(Transform):
    pass


class HeatEquation2D(PDE, LinearTransform):
    def __init__(self, latent):
        """
        latent must be a DifferentialOperatorJoint with a 2D differential kernel.

        The heat kernel transform is:
            df/dt - d^2f/dx^2 = 0
        """
        self._parent = latent
        self._output_dim = 1

    def mean(self, X):
        # parent is a DifferentialOperatorJoint whose outputs are:
        #   f, ft, ft2, fx, fx2 
        parent_mean = self.parent.mean_blocks(X)

        dt = parent_mean[0]
        dx2 = parent_mean[4]

        return dt - dx2

    def covar(self, X1, X2):

        parent_covar = self.parent.covar(X1, X2)

        N1 = X1.shape[0]
        N2 = X2.shape[0]

        left_idx = np.arange(N1)
        right_idx = np.arange(N2)

        Kt = parent_covar[left_idx+N1*1, :]
        Kt = Kt[:, right_idx+N2*1]

        Kx2 = parent_covar[left_idx+N1*4, :]
        Kx2 = Kx2[:, right_idx+N2*4]

        Ktx2 = parent_covar[left_idx+N1*1, :]
        Ktx2 = Ktx2[:, right_idx+N2*4]

        K =  Kt + Kx2 + 2 * Ktx2

        #ensure correct rank
        chex.assert_shape(K, [N1 * self.output_dim, N2 * self.output_dim]) 

        return K

