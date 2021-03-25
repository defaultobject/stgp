import jax.numpy as np
import objax

from . import ApproximatePosterior
from ..batching import batch
from ..computation.matrix_ops import vectorized_lower_triangular_cholesky, lower_triangle
import chex

class GaussianApproximatePosterior(ApproximatePosterior):
    def __init__(self, dim: int=None, m=None, S=None):

        if m is None:
            m = np.zeros([dim, 1])

        if S is None:
            S = np.eye(dim)

        if dim is None:
            dim = m.shape[0]

        self.raw_m = objax.TrainVar(m)
        self.raw_S_chol = objax.TrainVar(vectorized_lower_triangular_cholesky(S))

        self.dim = dim

    @batch
    def m(self, raw_fn):
        return raw_fn()

    @batch
    def S_chol(self, raw_fn):
        S_chol_raw = raw_fn()

        print(self)
        return lower_triangle(S_chol_raw, self.dim)

    @property
    def S(self):
        S_chol = self.S_chol
        return S_chol @ S_chol.T

    @property
    def S_diag(self):
        # TODO: implement in utils

        chol = self.S_chol
        return np.sum(np.square(chol), axis=0)[:, None]
        #return np.diag(self.S)[:, None]




