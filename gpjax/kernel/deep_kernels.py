from . import Kernel
from .kernel import StationaryKernel, ConcatationKernel
from ..dispatch import evoke

import jax
import jax.numpy as np
from typing import List, Optional, Union

import chex 

class DeepIndependentKernel(ConcatationKernel):
    def K(self, X1: np.array, X2: np.array):
        K_arr = super(DeepIndependentKernel, self).K(X1, X2)

        return np.prod(K_arr, axis=0)

    def K_diag(self, X1: np.array):
        K_arr = super(DeepIndependentKernel, self).K_diag(X1)
        return np.prod(K_arr, axis=0)

class DeepStationary(StationaryKernel):
    def _K(self, X1, X2):
        D = X1.shape[1]

        def _K_d2(x1, x2):
            #vectorised over 2nd input
            chex.assert_rank(x1, 1)
            chex.assert_rank(x2, 1)

            chex.assert_equal(x1.shape[0], D)
            chex.assert_equal(x2.shape[0], D)

        
            #a deep kernel is 1d only
            k_xx = self._K_scaler(x1, x2, self.variance, self.lengthscales[0])

            chex.assert_rank(k_xx, 0)

            return k_xx

        def _K_d1(x1, X2):
            #vectorised over first input
            return jax.vmap(_K_d2, in_axes=[None, 0], out_axes=0)(x1, X2)

        K = jax.vmap(_K_d1, in_axes=[0, None], out_axes=0)(X1, X2)

        chex.assert_equal(K.shape[0], X1.shape[0])
        chex.assert_equal(K.shape[1], X2.shape[0])

        return K

class DeepRBF(DeepStationary):
    def __init__(
        self, 
        kernel: Optional['Kernel'] = None,
        parent_model: Optional['Model'] = None,
        lengthscale: Optional[np.ndarray] = None, 
        variance: Optional[np.ndarray] = None, 
        input_dim: Optional[int] = 1, 
        active_dims: Optional[np.ndarray] = None
    ):

        super(DeepRBF, self).__init__(lengthscale, variance, input_dim, active_dims)

        self.parent_kernel = kernel
        self.parent_model = parent_model





    def _K_scaler(self, x1, x2, variance, lengthscale):
        #TODO: generalise to multi dimensions

        print(x1.shape)
        print(x2.shape)
        _x1 = np.reshape( x1, [1, -1])
        _x2 = np.reshape( x2, [1, -1])

        x_stacked = np.vstack([_x1, _x2])

        chex.assert_equal(x_stacked.shape[0], 2)

        if self.parent_model is not None:
            parent_mean, parent_k = self.parent_model.predict(x_stacked, diagonal=False)
        else:
            parent_mean = None
            parent_k = self.parent_kernel.K(x_stacked, x_stacked)

        chex.assert_rank(parent_k, 2)

        k_11 = parent_k[0, 0]
        k_12 = parent_k[0, 1]
        k_22 = parent_k[1, 1]
        
        if parent_mean is None:
            L = lengthscale + k_11 + k_22 - 2*k_12
            Kij =  np.sqrt(lengthscale)*variance / np.sqrt(L)
        else:
            L = lengthscale + k_11 + k_22 - 2*k_12
            Kij =  np.exp(-(1/(2*L))*(parent_mean[0]-parent_mean[1])**2)*np.sqrt(lengthscale)*variance / np.sqrt(L)


        return Kij

        #return variance * (1/np.sqrt(1 + (k_11 + k_22 - 2*k_12)/(lengthscale)))

        

