"""Base transform class."""
from ..utils.utils import ensure_module_list, can_batch, get_batch_type
from batchjax import batch_or_loop, BatchType

import jax
import jax.numpy as np
import objax
import chex

from typing import List, Optional


class Transform(objax.Module):
    """All transforms must be define a forward or inverse method."""
    def __init__(self):
        self._num_latents = None
        self._num_outputs = None

        self._output_dim = None
        self._input_dim = None

        self._latent_obj = None
        self._latents_arr = None

        self.batches = None

    def transform_diagonal(self, mu, var):
        """Transform a Gaussian dist """
        raise NotImplementedError()

    def forward(self, x):
        """Compute f=T(x)."""
        raise NotImplementedError()

    def inverse(self, f):
        """Compute x=T^{-1}(f)."""
        raise NotImplementedError()

    @property
    def num_outputs(self):
        return self._num_outputs

    @property
    def output_dim(self): return self._output_dim

    @property
    def input_dim(self): return self._input_dim

    def get_kernels(self):
        return objax.ModuleList([g.kernel for g in self.latents])

    @property
    def latents(self):
        return self._latents_arr

    @property
    def latent_obj(self):
        return self._latent_obj

    def get_batches(self):
        return self.batches

class NonLinearTransform(Transform):
    @property
    def num_latents(self):
        return len(self.latents)

    def get_sparsity_list(self):
        return [p.sparsity for p in self.latents]

    @property
    def latents(self):
        return self.latent_obj._latents_arr


class LinearTransform(Transform):
    """
    All linear transforms support .W returns the mixing matrix
    """

    def get_sparsity_list(self):
        return [p.sparsity for p in self.latents]

    @property
    def latents(self):
        return self.latent_obj._latents_arr

    @property
    def num_latents(self):
        return len(self.latent_obj.latents)

    def transform_diagonal(self, mu, var):
        W = self.W

        # Mixing latent functions
        mu = W @ mu[..., 0] 
        var = np.square(W) @ var[..., 0] 

        # fix shapes

        mu = mu[..., None]
        var = var[..., None]

        return mu, var

    def mean(self, X1):
        """ Output shape [P, N1]. """
        raise NotImplementedError()

    def vec_mean(self, X1: np.ndarray) -> np.ndarray:
        N1 = X1.shape[0]
        P = self.num_outputs

        mean = self.mean(X1)
        mean = np.vstack(mean)

        chex.assert_shape(mean, [N1*P, 1])
        return mean

    def covar(self, X1, X2):
        """ Output shape [P, N1, N2]. """
        raise NotImplementedError()

    def full_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        k_arr = self.covar(X1, X2)
        k =  jax.scipy.linalg.block_diag(*k_arr)

        chex.assert_shape(
            k,
            [self.num_latents*X1.shape[0], self.num_latents*X2.shape[0]]
        )

        return k

    def var(self, X1):
        """ Output shape [P, N1]. """
        raise NotImplementedError()

    def vec_var(self, X1: np.ndarray) -> np.ndarray:
        N1 = X1.shape[0]
        P = self.num_outputs
        var = self.var(X1)

        var = np.hstack(var)[:, None]

        chex.assert_shape(var, [N1*P, 1])
        return var

    def full_var(self, X1):
        """ Output shape [PxN1, PxN1]. """
        raise NotImplementedError()

class Independent(LinearTransform):
    def __init__(
        self, 
        latents: Optional[List['Model']] = None, 
        latent: Optional['Model'] = None,
        prior=True
    ) -> None:
        """
        Args:
            prior: bool -- Indicates whether latents are priors or posteriors
        """ 
        super().__init__()

        self.prior = prior

        if (latents is None) and (latent is None):
            raise RuntimeError('Latents must be passed')

        if (latent is not None) and (latents is not None):
            raise RuntimeError('Only latent or latents must be passed')

        if latent:
            # Standardize input to ease implementation
            latents = [latent]

        if prior:
            self._num_latents = len(latents)
        else:
            self._num_latents = latent.num_outputs


        self._latents_arr = ensure_module_list(latents)
        self._num_outputs = self.num_latents

    def forward(self, x):
        """Compute f=T(x)."""
        return x

    def inverse(self, f):
        """Compute x=T^{-1}(f)."""
        return f

    def get_Z(self):
        Z_arr = batch_or_loop(
            lambda latent:  latent.sparsity.Z,
            [self.latents],
            [0],
            dim = self.num_latents,
            out_dim = 1,
            batch_type = get_batch_type(self.latents)
        )

        return Z_arr

    @property
    def latent_obj(self):
        # For consistency with other transform classes
        return self 

    @property
    def num_latents(self):
        return len(self.latents)


    def mean(self, X1: np.ndarray) -> np.ndarray:
        mean = batch_or_loop(
            lambda X1, latent:  latent.mean(X1),
            [X1, self.latents],
            [None, 0],
            dim = self.num_latents,
            out_dim = 1,
            batch_type = BatchType.LOOP
        )

        mean = np.reshape(
            mean, 
            [self.num_outputs, X1.shape[0], 1]
        )

        return mean


    def covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        k_arr = batch_or_loop(
            lambda X1, X2, latent:  latent.covar(X1, X2),
            [X1, X2, self.latents],
            [None, None, 0],
            dim = self.num_latents,
            out_dim = 1,
            batch_type = BatchType.LOOP
        )

        k_arr = np.reshape(
            k_arr, 
            [self.num_outputs, X1.shape[0], X2.shape[0]]
        )

        return k_arr


    def var(self, X1: np.ndarray) -> np.ndarray:
        var = batch_or_loop(
            lambda X1, latent:  latent.var(X1),
            [X1, self.latents],
            [None, 0],
            dim = self.num_latents,
            out_dim = 1,
            batch_type = BatchType.LOOP
        )

        var = np.reshape(
            var, 
            [self.num_outputs, X1.shape[0]]
        )

        return var

    def full_var(self, X1: np.ndarray) -> np.ndarray:
        return self.covar(X1, X1)

class SumTransform(LinearTransform):
    def __init__(self, t1: Transform, t2: Transform):
        self.t1 = t1
        self.t2 = t2

        # Make sure t1 and t2 are compatable
        chex.assert_equal(
            self.t1.num_outputs,
            self.t2.num_outputs
        )

        self._num_outputs = self.t1.num_outputs
        self._num_latents = self.num_outputs

    def vec_mean(self, X1): 
        return self.t1.vec_mean(X1) + self.t2.vec_mean(X1)

    def mean(self, X1): 
        return self.t1.mean(X1) + self.t2.mean(X1)

    def covar(self, X1, X2): 
        return self.t1.covar(X1, X2) + self.t2.covar(X1, X2)

    def full_covar(self, X1, X2): 
        return self.t1.full_covar(X1, X2) + self.t2.full_covar(X1, X2)

    def var(self, X1): 
        return self.t1.var(X1) + self.t2.var(X1)

    def vec_var(self, X1): 
        return self.t1.vec_var(X1) + self.t2.vec_var(X1)

    def full_var(self, X1): 
        return self.t1.full_var(X1) + self.t2.full_var(X1)

class One2One(Independent):
    def __init__(self, in_model: Transform, out_models: List[Transform]):
        self.in_model = in_model
        self.out_models = ensure_module_list(out_models)

        self._output_dim = self.in_model.output_dim
        self._input_dim = self.output_dim

    def mean(self, X1): 
        m =  self.in_model.mean(X1)
        chex.assert_shape(m, [self.output_dim, X1.shape[0], 1])
        return m

    def covar(self, X1, X2): 
        P = self.output_dim
        N1 = X1.shape[0]
        N2 = X2.shape[0]

        # precompute kernels from in_model
        prior_covar = self.in_model.covar(X1, X2)
        prior_var_1 = self.in_model.var(X1)
        prior_var_2 = self.in_model.var(X2)
        prior_mean_1 = self.in_model.mean(X1)
        prior_mean_2 = self.in_model.mean(X2)

        def _propogate(X1, X2, model_p, mean_p_1, mean_p_2, prior_var_1, prior_var_2, covar_p):
            return model_p.kernel.forward(X1, X2, mean_p_1, mean_p_2, prior_var_1, prior_var_2, covar_p)

        # push each outputs covar through a kernel
        covar = batch_or_loop(
            _propogate,
            [X1, X2, self.out_models, prior_mean_1, prior_mean_2, prior_var_1, prior_var_2, prior_covar],
            [None, None, 0, 0, 0, 0, 0, 0],
            dim=self.output_dim,
            out_dim=1,
            batch_type = BatchType.LOOP
        )

        chex.assert_shape(covar, [P, N1, N2])
        return covar

    def var(self, X1): 
        P = self.output_dim
        N1 = X1.shape[0]

        # precompute input mean and variances
        prior_var = self.in_model.var(X1)
        prior_mean_1 = self.in_model.mean(X1)

        def _propogate(X1, model_p, mean_p_1, prior_var_p):
            # TODO: model_p should just be a deep kernel
            return model_p.kernel.forward_diag(X1, mean_p_1, prior_var_p)

        # push each outputs covar through a kernel
        var = batch_or_loop(
            _propogate,
            [X1, self.out_models, prior_mean_1, prior_var],
            [None, 0, 0, 0, 0],
            dim=self.output_dim,
            out_dim=1,
            batch_type = BatchType.LOOP
        )

        chex.assert_shape(var, [P, N1])
        return var




class ElementWiseTransform(Transform):
    def __init__(self):
        super(ElementWiseTransform, self).__init__()
        self._num_latents = 1
        self._num_outputs = 1
