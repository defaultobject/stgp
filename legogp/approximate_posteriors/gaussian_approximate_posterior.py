import jax.numpy as np
import objax

from . import ApproximatePosterior
from ..computation.matrix_ops import vectorized_lower_triangular_cholesky, lower_triangle, diagonal_from_cholesky
import chex
import warnings

class GaussianApproximatePosterior(ApproximatePosterior):
    def __init__(self, dim: int=None, m=None, S=None):
        super(GaussianApproximatePosterior, self).__init__()

        if dim is None and m is None:
            raise RuntimeError('Either dim or m must be passed')

        if m is None:
            m = 0.01*np.ones([dim, 1])

        if S is None:
            warnings.warn('Approximate posterior ')
            S = 0.1*np.eye(dim)

        if dim is None:
            dim = m.shape[0]

        self._m = objax.TrainVar(m)

        self._S_chol = objax.TrainVar(
            vectorized_lower_triangular_cholesky(S)
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
