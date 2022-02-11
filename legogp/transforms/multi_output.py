"""Multi-output/task specific transforms."""
from .transform import Transform, LinearTransform, NonLinearTransform, Independent
import typing
from typing import List, Optional, Union
import jax
import jax.numpy as np
import numpy as onp
import objax
import chex
from ..batching import batch
from ..computation.parameter_transforms import get_correlation_cholesky, correlation_transform
from ..computation.parameter_transforms import inv_positive_transform, positive_transform

class LMC_Base(LinearTransform):
    """
    Inherits
        num_outputs
        num_latents
    """
    def __init__(self, latents, input_dim: int, output_dim: int):

        super(LinearTransform, self).__init__()

        # Allow passing a list of prior models and transformed model
        if type(latents) is list:
            self._latents = Independent(latents=latents, prior=True)
        else:
            self._latents = latents

        self._input_dim = input_dim
        self._output_dim = output_dim

    @property
    def W(self):
        raise NotImplementedError()

    def vec_mean(self, X1: np.ndarray) -> np.ndarray:
        N1 = X1.shape[0]
        P = self.output_dim

        mean = np.zeros([P, N1])

        mean = np.hstack(mean)[:, None]
        chex.assert_shape(mean, [P*N1, 1])

        return mean

    def full_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        N1 = X1.shape[0]
        N2 = X2.shape[0]

        mixing_matrix = self.W

        Q = self.input_dim
        P = self.output_dim

        W1 = np.kron(mixing_matrix, np.eye(N1))
        W2 = np.kron(mixing_matrix, np.eye(N2))

        K_bdiag = self.latents.full_covar(X1, X2)
        chex.assert_shape(K_bdiag, [Q*N1, Q*N2])

        covar = W1 @ K_bdiag @ W2.T
        chex.assert_shape(covar, [P*N1, P*N2])

        return covar

    def vec_var(self, X1: np.ndarray) -> np.ndarray:
        N1 = X1.shape[0]
        Q = self.input_dim
        P = self.output_dim

        mixing_matrix = self.W

        K_diag = self.latents.var(X1)
        chex.assert_shape(K_diag, [Q, N1])

        W = mixing_matrix**2

        #P x N
        covar = W @ K_diag
        chex.assert_shape(covar, [P, N1])

        # PN
        covar = np.hstack(covar)[:, None]
        chex.assert_shape(covar, [P*N1, 1])

        return covar

class LMC_Unit_Tri(LMC_Base):
    def __init__(
        self, 
        latents: Optional[Union[List['Model'], Transform]]=None, 
        output_dim: Optional[int]=None, 
        input_dim: Optional[int]=None, 
        W: Optional[np.ndarray] = None
    ):
        super().__init__(latents, input_dim=latents.num_latents, output_dim=output_dim)

        self._num_latents = self.input_dim

        # Setup correlation matrix variables
        num_vars = int(self.output_dim*(self.output_dim-1)/2)
        self.z_arr = objax.TrainVar(np.zeros(num_vars))
        #self.z_arr = objax.StateVar(onp.zeros(num_vars))

    @property
    def W(self):
        P = self.output_dim
        Q = self.input_dim

        tri = np.eye(P, Q)
        mixing_matrix = tri.at[jax.ops.index[np.tril_indices(P, -1, Q)]].set(self.z_arr.value)

        return mixing_matrix
