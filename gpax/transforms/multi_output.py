"""Multi-output/task specific transforms."""
from .transform import Transform, LinearTransform, NonLinearTransform, Independent
import typing
from typing import List, Optional
import jax
import jax.numpy as np
import numpy as onp
import objax
import chex
from ..batching import batch
from ..computation.parameter_transforms import get_correlation_cholesky, correlation_transform
from ..computation.parameter_transforms import inv_positive_transform, positive_transform

class LMC(LinearTransform):
    """
    Inherits
        num_outputs
        num_latents
    """
    def __init__(self):
        super().__init__()

    @property
    def W(self):
        raise NotImplementedError()

    def vec_mean(self, X1: np.ndarray) -> np.ndarray:
        N1 = X1.shape[0]
        P = self.num_outputs

        mean = np.zeros([P, N1])

        mean = np.hstack(mean)[:, None]
        chex.assert_shape(mean, [P*N1, 1])

        return mean

    def full_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        N1 = X1.shape[0]
        N2 = X2.shape[0]

        mixing_matrix = self.W

        Q = self.num_latents
        P = self.num_outputs

        W1 = np.kron(mixing_matrix, np.eye(N1))
        W2 = np.kron(mixing_matrix, np.eye(N2))

        K_bdiag = self.latents.full_covar(X1, X2)
        chex.assert_shape(K_bdiag, [Q*N1, Q*N2])

        covar = W1 @ K_bdiag @ W2.T
        chex.assert_shape(covar, [P*N1, P*N2])

        return covar

    def vec_var(self, X1: np.ndarray) -> np.ndarray:
        N1 = X1.shape[0]
        Q = self.num_latents
        P = self.num_outputs

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

class LMC_Corr_var(LinearTransform):
    def __init__(self, latents: Optional[List['Model']]=None, output_dim: Optional[int]=None, input_dim: Optional[int]=None, W: Optional[np.ndarray] = None):
        super(LMC_Corr_var, self).__init__()

        if input_dim is None:
            input_dim = len(latents)

        self._latents = objax.ModuleList(latents)
        self.output_dim = output_dim

        self._num_outputs = output_dim
        self.input_dim = input_dim

        num_vars = int(self.output_dim*(self.output_dim-1)/2)
        self.delta_arr = objax.TrainVar(onp.zeros(num_vars))
        self.var = objax.TrainVar(inv_positive_transform(onp.ones(output_dim)))

    @property
    def W(self):
        P = self.output_dim
        Q = self.input_dim

        z_arr = correlation_transform(self.delta_arr.value, 1.0)
        mixing_matrix = get_correlation_cholesky(z_arr, P, Q)

        mixing_matrix = np.diag(positive_transform(self.var.value)) @ mixing_matrix

        return mixing_matrix

class LMC_Corr(LinearTransform):
    def __init__(self, latents: Optional[List['Model']]=None, output_dim: Optional[int]=None, input_dim: Optional[int]=None, W: Optional[np.ndarray] = None):
        super(LMC_Corr, self).__init__()

        if input_dim is None:
            input_dim = len(latents)

        self._latents = objax.ModuleList(latents)
        self.output_dim = output_dim

        self._num_outputs = output_dim
        self.input_dim = input_dim

        num_vars = int(self.output_dim*(self.output_dim-1)/2)
        self.delta_arr = objax.TrainVar(onp.zeros(num_vars))

    @property
    def W(self):
        P = self.output_dim
        Q = self.input_dim

        z_arr = correlation_transform(self.delta_arr.value, 1.0)
        mixing_matrix = get_correlation_cholesky(z_arr, P, Q)

        return mixing_matrix

class LMC_Unit_Tri(LMC):
    def __init__(self, latents: Optional[List['Model']]=None, output_dim: Optional[int]=None, input_dim: Optional[int]=None, W: Optional[np.ndarray] = None):
        super().__init__()

        self._latents = Independent(latents=latents, prior=True)
        input_dim = self.latents.num_latents
        self._num_latents = input_dim

        self.output_dim = output_dim

        self._num_outputs = output_dim
        self.input_dim = input_dim

        num_vars = int(self.output_dim*(self.output_dim-1)/2)
        self.z_arr = objax.TrainVar(onp.zeros(num_vars))

    @property
    def W(self):
        P = self.output_dim
        Q = self.input_dim

        tri = np.eye(P, Q)
        mixing_matrix = tri.at[jax.ops.index[np.tril_indices(P, -1, Q)]].set(self.z_arr.value)

        return mixing_matrix


class _LMC(LinearTransform):
    r"""
    Linear model of coregionilisation.

    Generative model:
        f_p = \sum w_{p,q} g_q

    """
    class LMC_p(LinearTransform):
        """
            Individual transform for task p
        """
        def __init__(self, latents_p, raw_W, p):
            #make a list so that objax does not try to expand it

            self.raw_W = raw_W
            self.latents_p = latents_p
            self.p = p 

        @property
        def W_p(self):
            return self.raw_W.value[self.p]

        def forward(self, latent_means):

            return np.sum(latent_means * self.W_p[:, None, None], axis=0)


    def __init__(self, latents: Optional[List['Model']]=None, output_dim: Optional[int]=None, input_dim: Optional[int]=None, W: Optional[np.ndarray] = None):
        print('CREATING')
        super(LMC, self).__init__()

        if input_dim is None:
            input_dim = len(latents)

        self._latents = objax.ModuleList(latents)
        self.output_dim = output_dim

        self._num_outputs = output_dim

        self.input_dim = input_dim

        self.raw_W = objax.TrainVar(np.eye(self.output_dim, self.input_dim))

        #self.batches = objax.ModuleList([LMC.LMC_p(self.latents, self, p) for p in range(self.output_dim)])
        self.batches = [LMC.LMC_p(self._latents, objax.TrainRef(self.raw_W), p) for p in range(self.output_dim)]


    @batch
    def W(self, raw_getter):
        return raw_getter()

    def _W(self):
        W = []
        for b in self.batches:
            W.append(b.W_p)

        W =  np.array(W)

        chex.assert_equal(W.shape, (self.output_dim, self.input_dim))

        return W


class GPRN(NonLinearTransform):
    r"""
    Gaussian Process Regression Network.

    Generative model:
        f_p = \sum w_{p,q}(x) g_q(x)
    """
    class GPRN_p(NonLinearTransform):
        """
            Individual transform for task p
        """
        def __init__(self, latents_f, latents_w_p):
            self.latents_f = latents_f
            self.latents_w_p = latents_w_p

        def forward(self, latent_means):
            return np.sum(latent_means, axis=0)

    def __init__(self, latent_f: Optional[List['Model']]=None, latent_w: Optional[List[List['Model']]]=None, output_dim: Optional[int]=None, input_dim: Optional[int]=None):
        self.latent_f = objax.ModuleList(latent_f)
        self.latent_w = objax.ModuleList(latent_w)
        self.output_dim = len(self.latent_w)
        self.input_dim = len(self.latent_f)

        self.batches = objax.ModuleList([GPRN.GPRN_p(self.latent_f, self.latent_w[p]) for p in range(self.output_dim)])

    def number_of_latents(self):
        return self.input_dim + self.input_dim * self.output_dim

    @property
    def latents(self):
        latent_W = [self.latent_w[p][q] for p in range(self.output_dim) for q in range(self.input_dim)]
        latent_f = [self.latent_f[q]  for q in range(self.input_dim)]
        return latent_f + latent_W

class ConstrainedLMC(Transform):
    r"""
    Constrained Linear model of coregionilisation.

    Generative model:
        f_p = \sum w_{p,q} g_q
    """


class ConstrainedGPRN(Transform):
    r"""
    Constrained Gaussian Process Regression Network.

    Generative model:
        f_p = \sum w_{p,q}(x) g_q(x)
    """
