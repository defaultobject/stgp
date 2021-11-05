"""Base transform class."""
from ..utils.utils import ensure_module_list
from ..batching import loop_or_batch

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

    def vec_mean(self, X1):
        """ Output shape [PxN1, 1]. """
        raise NotImplementedError()

    def covar(self, X1, X2):
        """ Output shape [P, N1, N2]. """
        raise NotImplementedError()

    def full_covar(self, X1, X2):
        """ Output shape [PxN1, PxN2]. """
        raise NotImplementedError()

    def var(self, X1):
        """ Output shape [P, N1]. """
        raise NotImplementedError()

    def vec_var(self, X1):
        """ Output shape [PxN1, 1]. """
        raise NotImplementedError()

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
        if self.prior:
            # Assume that latents are zero mean
            mean = np.zeros([self.num_latents, X1.shape[0]])
        else:
            mean, _ = self.latents[0].predict(X1, diagonal=True)

        chex.assert_shape(
            mean, 
            [self.num_outputs, X1.shape[0]]
        )
        return mean

    def vec_mean(self, X1: np.ndarray) -> np.ndarray:
        N1 = X1.shape[0]
        P = self.num_outputs

        mean = self.mean(X1)
        mean = np.hstack(mean)[:, None]

        chex.assert_shape(mean, [N1*P, 1])
        return mean

    def covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.prior:
            k_arr = loop_or_batch(
                lambda X1, X2, latent:  latent.kernel[0].K(X1, X2),
                [X1, X2, self.latents],
                [None, None, 0],
                self.num_latents,
                num_returned_args = 1
            )

            return k_arr

        else:
            k_arr = self.latents[0].predictive_covar(X1, X2)

        chex.assert_shape(
            k_arr, 
            [self.num_latents, X1.shape[0], X2.shape[0]]
        )

        return k_arr

    def full_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        k_arr = self.covar(X1, X2)
        k =  jax.scipy.linalg.block_diag(*k_arr)

        chex.assert_shape(
            k,
            [self.num_latents*X1.shape[0], self.num_latents*X2.shape[0]]
        )

        return k

    def var(self, X1: np.ndarray) -> np.ndarray:
        if self.prior:
            var = loop_or_batch(
                lambda X1, latent:  latent.kernel[0].K_diag(X1),
                [X1, self.latents],
                [None, 0],
                self.num_latents,
                num_returned_args = 1
            )


        else:
            # for posterior we only support one latent
            _, var =  self.latents[0].predict(X1, diagonal=True)

        chex.assert_shape(
            var, 
            [self.num_latents, X1.shape[0]]
        )

        return var

    def vec_var(self, X1: np.ndarray) -> np.ndarray:
        N1 = X1.shape[0]
        P = self.num_outputs
        var = self.var(X1)

        var = np.hstack(var)[:, None]

        chex.assert_shape(var, [N1*P, 1])
        return var

    def full_var(self, X1: np.ndarray) -> np.ndarray:
        N1 = X1.shape[0]
        P = self.num_outputs

        if self.prior:
            var = self.full_covar(X1, X1)
        else:
            # for posterior we only support one 
            _, var =  self.latents[0].predict(X1, diagonal=False)

        chex.assert_shape(var, [P*N1, P*N1])
        return var

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

class NonLinearTransform(Transform):
    pass

class ElementWiseTransform(Transform):
    def __init__(self):
        super(ElementWiseTransform, self).__init__()
        self._num_latents = 1
        self._num_outputs = 1
