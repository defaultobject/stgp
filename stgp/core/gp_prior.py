from .models import Prior
from ..kernels import RBF
from ..sparsity import NoSparsity

import jax
import jax.numpy as np
import objax
from typing import Optional
import warnings

import chex

class GPPrior(Prior):
    def __init__(
        self, 
        X: np.ndarray = None, 
        kernel: Optional['Kernel'] = None,
        sparsity: Optional['Sparsity'] = None,
        **kwargs
    ):
        super(GPPrior, self).__init__()

        self._kernel = kernel
        self.sparsity = sparsity
        self.set_defaults()

    def set_defaults(self):
        if self.kernel is None:
            raise NotImplementedError()
            warnings.warn(f'Using default ARD RBF kernel with input dim {self.input_dim} ')
            self._kernel = RBF(
                lengthscales=[1.0 for d in range(self.input_dim)],
                input_dim=self.input_dim
            )

        if self.sparsity is None:
            raise NotImplementedError()
            self.sparsity = NoSparsity(self.X)

    @property
    def latent_obj(self):
        return self

    @property
    def X(self): return self._X.value

    @property
    def kernel(self): return self._kernel

    @property
    def input_dim(self): return self.X.shape[1]

    @property
    def output_dim(self): return 1

    def mean(self, XS):
        """ Assume a zero mean GP """
        return np.zeros(XS.shape[0])[None, :, None]

    def var(self, XS):
        k =  self.kernel.K_diag(XS)
        chex.assert_shape(k, [XS.shape[0]])
        return k[..., None]

    def var_blocks(self, XS):
        k =  self.kernel.K_diag(XS)
        k = k[None, :]
        chex.assert_shape(k, [1, XS.shape[0]])
        return k[..., None]

    def covar_blocks(self, X1, X2):
        k = self.kernel.K(X1, X2)
        k = k[None, :]
        chex.assert_shape(k, [1, X1.shape[0], X2.shape[0]])
        return k

    def covar(self, X1, X2):
        k = self.kernel.K(X1, X2)
        chex.assert_shape(k, [X1.shape[0], X2.shape[0]])
        return k


    def full_var(self, X):
        return self.covar(X, X)

    def sample(self, X1, X2):
        raise NotImplementedError()

