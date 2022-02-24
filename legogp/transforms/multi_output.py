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
from ..computation.parameter_transforms import get_correlation_cholesky, correlation_transform, inv_correlation_transform
from .. import Parameter

class GPRN_Base(NonLinearTransform):
    def __init__(self, W, f, input_dim: int = None, output_dim: int = None):

        super(NonLinearTransform, self).__init__()

        # Flatten W into a vector - row major ordering
        W_vec = [w for W_p in W for w in W_p]

        # Flatten latents to fit into VI framework
        self._latents = Independent(
            latents = f+W_vec,
            prior = True
        )

        self._input_dim = len(f)
        self._output_dim = len(W)

    @property
    def forward(self, f):
        raise NotImplementedError()

class GPRN(GPRN_Base):
    def forward(self, f):
        # TODO: 
        # f has the same ordering as self.latents
        latent_f = f[:self.input_dim]
        latent_W = f[self.input_dim:]

        #return latent_W[:self.input_dim]
        return latent_f

        # W is in row-major ordering
        latent_W = latent_W.reshape(
            self.output_dim,
            self.input_dim,
            order='C'
        )

        return latent_W @ latent_f

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

# TODO: implement ICM

class LMC(LMC_Base):
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
        self._W = Parameter(np.eye(self.output_dim, self.input_dim), name='W')

    @property
    def W(self):
        return self._W.value

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
        self.z_arr = Parameter(np.zeros(num_vars), name='LMC_Unit_Tri/Z_arr')

    @property
    def W(self):
        P = self.output_dim
        Q = self.input_dim

        tri = np.eye(P, Q)
        mixing_matrix = tri.at[jax.ops.index[np.tril_indices(P, -1, Q)]].set(self.z_arr.value)

        return mixing_matrix


class LMC_Corr(LMC_Base):
    def __init__(
        self, 
        latents: Optional[Union[List['Model'], Transform]]=None, 
        output_dim: Optional[int]=None, 
        variances: Optional[np.ndarray] = None,
        mixing_weights: Optional[np.ndarray] = None,
        a: Optional[float] = None
    ):
        super().__init__(latents, input_dim=latents.num_latents, output_dim=output_dim)

        self._num_latents = self.input_dim

        # When using LMC_corr the mixing matrix must be square
        self.P = self.output_dim
        self.Q = int(self.P*(self.P-1)/2)

        # Set defaults
        if variances is None:
            variances = np.ones(self.P)

        if mixing_weights is None:
            mixing_weights = np.zeros(self.Q)

        if a is None:
            self.a = 1.0

        # Setup Parameters
        self.variances = Parameter(variances, constraint='positive', name='variance')

        self.z_arr = Parameter(
            mixing_weights,
            constraint_fn=lambda x: correlation_transform(x, self.a), 
            inv_constraint_fn=lambda x: inv_correlation_transform(x, self.a), 
            name='LMC_Corr/z_arr'
        )
        

    @property
    def W(self):
        z_arr = self.z_arr.value
        correlation_cholesky =  get_correlation_cholesky(z_arr, self.P, self.Q)

        var_diag = np.diag(self.variances.value)

        return var_diag @ correlation_cholesky
