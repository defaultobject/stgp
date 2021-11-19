"""Base transform class."""
from ..utils.utils import ensure_module_list, can_batch
from batchjax import batch_or_loop

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
        self_latents = None
        self.batches = None

    def forward(self):
        """Compute f=T(x)."""
        pass

    def inverse(self):
        """Compute x=T^{-1}(f)."""
        pass

    @property
    def num_latents(self):
        return self._num_latents

    @property
    def num_outputs(self):
        return self._num_outputs

    def get_kernels(self):
        return objax.ModuleList([g.kernel for g in self.latents])

    @property
    def latents(self):
        return self._latents

    def get_batches(self):
        return self.batches





class LinearTransform(Transform):
    """
    All linear transforms support .W returns the mixing matrix
    """
    def mean(self, X1):
        """ Output shape [P, N1]. """
        raise NotImplementedError()

    def vec_mean(self, X1: np.ndarray) -> np.ndarray:
        N1 = X1.shape[0]
        P = self.num_outputs

        mean = self.mean(X1)
        mean = np.hstack(mean)[:, None]

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

        self._num_outputs = self.num_latents

        self._latents = ensure_module_list(latents)


    def mean(self, X1: np.ndarray) -> np.ndarray:
        mean = batch_or_loop(
            lambda X1, latent:  latent.mean(X1),
            [X1, self.latents],
            [None, 0],
            dim = self.num_latents,
            out_dim = 1,
            batch_flag = can_batch(self.latents)
        )

        chex.assert_shape(
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
            batch_flag = can_batch(self.latents)
        )

        chex.assert_shape(
            k_arr, 
            [self.num_latents, X1.shape[0], X2.shape[0]]
        )

        return k_arr


    def var(self, X1: np.ndarray) -> np.ndarray:
        var = batch_or_loop(
            lambda X1, latent:  latent.var(X1),
            [X1, self.latents],
            [None, 0],
            dim = self.num_latents,
            out_dim = 1,
            batch_flag = can_batch(self.latents)
        )

        chex.assert_shape(
            var, 
            [self.num_latents, X1.shape[0]]
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

class DeepKernel_One2One(Independent):
    def __init__(self, prior: Transform, kernels: List['DeepKernel']):
        self.prior = prior
        self.kernels = ensure_module_list(kernels)
        self._num_outputs = prior.num_outputs
        self._num_latents = self.num_outputs

    def mean(self, X1): 
        P = self.num_outputs
        N1 = X1.shape[0]
        return np.zeros([P, N1])

    def covar(self, X1, X2): 
        P = self.num_outputs
        N1 = X1.shape[0]
        N2 = X2.shape[0]
        prior_covar = self.prior.covar(X1, X2)
        prior_var_1 = self.prior.var(X1)
        prior_var_2 = self.prior.var(X2)
        prior_mean_1 = self.prior.mean(X1)
        prior_mean_2 = self.prior.mean(X2)

        # push each outputs covar through a kernel

        covar = loop_or_batch(
            lambda X1, X2, kernel_p, mean_p_1, mean_p_2, prior_var_1, prior_var_2, covar_p:  kernel_p.forward(X1, X2, mean_p_1, mean_p_2, prior_var_1, prior_var_2, covar_p),
            [X1, X2, self.kernels, prior_mean_1, prior_mean_2, prior_var_1, prior_var_2, prior_covar],
            [None, None, 0, 0, 0, 0, 0, 0],
            self.num_outputs,
            num_returned_args=1
        )

        chex.assert_shape(covar, [P, N1, N2])
        return covar



    def var(self, X1): 
        P = self.num_outputs
        N1 = X1.shape[0]
        prior_var = self.prior.var(X1)
        prior_mean_1 = self.prior.mean(X1)

        # push each outputs covar through a kernel

        var = loop_or_batch(
            lambda X1, kernel_p, mean_p_1, prior_var_p:  kernel_p.forward_diag(X1, mean_p_1, prior_var_p),
            [X1, self.kernels, prior_mean_1, prior_var],
            [None, 0, 0, 0, 0],
            self.num_outputs,
            num_returned_args=1
        )

        chex.assert_shape(var, [P, N1])
        return var



class NonLinearTransform(Transform):
    pass

class ElementWiseTransform(Transform):
    def __init__(self):
        super(ElementWiseTransform, self).__init__()
        self._num_latents = 1
        self._num_outputs = 1
