import jax.numpy as np
import objax

from . import ApproximatePosterior
from ..computation.matrix_ops import lower_triangular_cholesky, lower_triangle, diagonal_from_cholesky
from .. import Parameter
import chex
import warnings

class GaussianApproximatePosterior(ApproximatePosterior):
    def __init__(self, dim: int=None, m=None, S=None, S_chol_vec = None, train=True):
        super(GaussianApproximatePosterior, self).__init__()

        if dim is None and m is None:
            raise RuntimeError('Either dim or m must be passed')

        if m is None:
            m = 0.01*np.ones([dim, 1])
            #m = np.ones([dim, 1])

        chex.assert_rank(m, 2)

        if S is None and S_chol_vec is None:
            warnings.warn('Approximate posterior ')
            S = 0.1*np.eye(dim)

        if dim is None:
            dim = m.shape[0]

        self._m = Parameter(
            np.array(m),
            constraint=None,
            name='GaussianApproxPosterior/m',
            train=train
        )

        if S_chol_vec is None:
            S_chol_vec = lower_triangular_cholesky(S)

        self._S_chol = Parameter(
            np.array(S_chol_vec),
            constraint=None,
            name='GaussianApproxPosterior/S_chol',
            train=train
        )

        self.dim = dim

    @property
    def m(self):
        return self._m.value

    @property
    def S_chol(self):
        S_chol_raw = self._S_chol.value

        return lower_triangle(S_chol_raw, self.dim)

    @property
    def S(self):
        S_chol = self.S_chol
        return S_chol @ S_chol.T

    @property
    def S_diag(self):
        return diagonal_from_cholesky(self.S_chol)

    def fix(self):
        self._m.fix()
        self._S_chol.fix()

class DiagonalGaussianApproximatePosterior(GaussianApproximatePosterior):
    def __init__(self, dim: int=None, m=None, S_diag=None, train=True):
        self._m = Parameter(
            m,
            constraint=None,
            name='GaussianApproxPosterior/m',
            train=train
        )

        self._S_diag = Parameter(
            S_diag,
            constraint=None,
            name='GaussianApproxPosterior/S_diag',
            train=train
        )

    @property
    def m(self):
        return self._m.value

    @property
    def S_diag(self):
        return self._S_diag.value





class FullGaussianApproximatePosterior(GaussianApproximatePosterior):
    """ Gaussian approximate posterior defined in latent-data format """
    pass

class DataLatentBlockDiagonalApproximatePosterior(FullGaussianApproximatePosterior):
    """ 
    Block diagonal approximate posterior that is parameterised in data-latent order.
    This inherits FullGaussianApproximatePosterior as it is used by CVI in the FullGaussianApproximatePosterior
    """
    def __init__(self,  m=None, S_blocks=None, train=True):
        self._m = Parameter(
            m,
            constraint=None,
            name='GaussianApproxPosterior/m',
            train=train
        )

        self._S_blocks = Parameter(
            S_blocks,
            constraint=None,
            name='GaussianApproxPosterior/S_blocks',
            train=train
        )

    @property
    def m(self):
        return self._m.value

    @property
    def S_blocks(self):
        return self._S_blocks.value
