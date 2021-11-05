from . import Kernel
from .kernel import StationaryKernel, ConcatationKernel
from ..dispatch import evoke
from .. import settings

import jax
import jax.numpy as np

from jax.scipy.special import erf

from typing import List, Optional, Union

import chex 
from ..batching import batch

class DeepIndependentKernel(ConcatationKernel):
    def K(self, X1: np.array, X2: np.array):
        K_arr = super(DeepIndependentKernel, self).K(X1, X2)

        return np.sum(K_arr, axis=0)

    def K_diag(self, X1: np.array):
        K_arr = super(DeepIndependentKernel, self).K_diag(X1)
        return np.sum(K_arr, axis=0)

class DeepNN(Kernel):
    def __init__(self, kernel, nn):
        self.kernel = kernel
        self.nn = nn

    def K_diag(self, X):
        X_nn = self.nn(X)
        return self.kernel.K_diag(X_nn)

    def K(self, X1, X2):
        X1_nn = self.nn(X1)
        X2_nn = self.nn(X2)
        return self.kernel.K(X1_nn, X2_nn)

class DeepHetreo(Kernel):
    def __init__(
        self,
        parent_model: Optional['Model'] = None,
        ignore_var = False
    ):
        self.parent_model = parent_model
        self.ignore_var = ignore_var

    def propogate_parent(self, X):
        parent_mean, parent_k = self.parent_model.predict(X, diagonal=True)

        return parent_mean, parent_k

    def K_diag(self, X):
        pm, pk = self.propogate_parent(X)

        if self.ignore_var:
            k =  np.exp(pm)
        else:
            k =  np.exp(pm+pk/2)

        return k

    def K(self, X1, X2):
        pm, pk = self.propogate_parent(X1)

        pm = np.tile(pm[:, None], [1, X2.shape[0]])
        pk = np.tile(pk[:, None], [1, X2.shape[0]])

        wn_kern = ((X1-X2.T)==0).astype(float)

        pm = pm *wn_kern
        pk = pk *wn_kern

        if self.ignore_var:
            k = np.exp(pm)*wn_kern
        else:
            k = np.exp(pm + pk/2)*wn_kern

        chex.assert_shape(k, [X1.shape[0], X2.shape[0]])

        return k



class DeepStationary(StationaryKernel):
    def __init__(
        self, 
        kernel: Optional['Kernel'] = None,
        parent_model: Optional['Model'] = None,
        lengthscale: Optional[np.ndarray] = None, 
        variance: Optional[np.ndarray] = None, 
        input_dim: Optional[int] = 1, 
        active_dims: Optional[np.ndarray] = None,
        use_X=True
    ):


        self.parent_kernel = kernel
        self.parent_model = parent_model
        self.use_X = use_X

        input_dim = 1
        if self.parent_model is not None:
            input_dim = self.parent_model.num_outputs

        super(DeepStationary, self).__init__(lengthscale, variance, input_dim, active_dims)


    def forward(self, X1, X2, mu_1, mu_2, K):
        # TODO: implement
        return K

    def forward_diag(self, X1, mu_1, K_diag):
        # TODO: implement
        return K_diag

    def propogate_parent(self, x1, x2):
        #_x1 = np.reshape( x1, [1, -1])
        #_x2 = np.reshape( x2, [1, -1])

        _x1 = x1
        _x2 = x2

        x_stacked = np.vstack([_x1, _x2])

        chex.assert_rank(x_stacked, 2)

        if self.parent_model is not None:
            parent_mean, parent_k = self.parent_model.predict(x_stacked, diagonal=False)
        else:
            parent_mean = np.zeros(x_stacked.shape[0])
            parent_k = self.parent_kernel.K(x_stacked, x_stacked)

        return parent_mean, parent_k

    def K_diag(self, X1):
        return self.variance * np.ones(X1.shape[0])

    def _K(self, X1, X2):

        parent_mean, parent_k = self.propogate_parent(X1, X2)

        breakpoint()

        if not self.use_X:
            X1 = X1[:, 0][:, None]
            X2 = X2[:, 0][:, None]

        D = X1.shape[1]

        chex.assert_rank(parent_mean, 1)
        chex.assert_rank(parent_k, 2)

        # Get predictions for X1 and X2
        pm_x1 = parent_mean[:X1.shape[0]]
        pm_x2 = parent_mean[X1.shape[0]:]

        # Get joint_covariances
        pk_x1x1 = np.diag(parent_k[:X1.shape[0], :X1.shape[0]])
        pk_x2x2 = np.diag(parent_k[X1.shape[0]:, X1.shape[0]:])
        pk_x1x2 = parent_k[:X1.shape[0], X1.shape[0]:]

        def _K_d2(x1, x2, pm_x1, pm_x2, pk_x1x1, pk_x2x2, pk_x1x2):
            #vectorised over 2nd input
            chex.assert_rank(x1, 1)
            chex.assert_rank(x2, 1)

            chex.assert_equal(x1.shape[0], D)
            chex.assert_equal(x2.shape[0], D)

            #vectorise over input dim
            k_d1_d2 = jax.vmap(self._K_scaler, in_axes=[0, 0, None, 0, None, None, None, None, None])(x1, x2, self.variance, self.lengthscales, pm_x1, pm_x2, pk_x1x1, pk_x2x2, pk_x1x2)

            chex.assert_equal(k_d1_d2.shape[0], D)

            k_xx =  np.product(k_d1_d2)

            chex.assert_rank(k_xx, 0)

            return k_xx

        def _K_d1(x1, X2, pm_x1, pm_x2, pk_x1x1, pk_x2x2, pk_x1x2):
            #vectorised over first input
            return jax.vmap(_K_d2, in_axes=[None, 0, None, 0, None, 0, 0], out_axes=0)(x1, X2, pm_x1, pm_x2,  pk_x1x1, pk_x2x2, pk_x1x2)

        K = jax.vmap(_K_d1, in_axes=[0, None, 0, None, 0, None, 0], out_axes=0)(X1, X2, pm_x1, pm_x2, pk_x1x1, pk_x2x2, pk_x1x2)

        chex.assert_equal(K.shape[0], X1.shape[0])
        chex.assert_equal(K.shape[1], X2.shape[0])

        return K

class DeepRBF(DeepStationary):
    def _K_scaler(self, x1, x2, variance, lengthscale, m1, m2, k_11, k_22, k_12):
        #TODO: generalise to multi dimensions
        _x1 = np.reshape( x1, [1, -1])
        _x2 = np.reshape( x2, [1, -1])

        x_stacked = np.vstack([_x1, _x2])

        chex.assert_equal(x_stacked.shape[0], 2)

        if m1 is None:
            L = lengthscale + k_11 + k_22 - 2*k_12
            k_ij =  np.sqrt(lengthscale)*variance / np.sqrt(L)
        else:
            L = lengthscale + k_11 + k_22 - 2*k_12
            k_ij =  np.exp(-(1/(2*L))*(m1-m2)**2)*np.sqrt(lengthscale)*variance / np.sqrt(L)


        return k_ij

        
class DeepMatern12(DeepStationary):
    def __init__(
        self, 
        kernel: Optional['Kernel'] = None,
        parent_model: Optional['Model'] = None,
        lengthscale: Optional[np.ndarray] = None, 
        variance: Optional[np.ndarray] = None, 
        input_dim: Optional[int] = 1, 
        active_dims: Optional[np.ndarray] = None
    ):

        super(DeepMatern12, self).__init__(lengthscale, variance, input_dim, active_dims)

        self.parent_kernel = kernel
        self.parent_model = parent_model


    def _K_scaler(self, x1, x2, variance, lengthscale):
        #TODO: generalise to multi dimensions

        parent_mean, parent_k = self.propogate_parent(x1, x2)

        chex.assert_rank(parent_k, 2)

        k_11 = parent_k[0, 0]
        k_12 = parent_k[0, 1]
        k_22 = parent_k[1, 1]
        
        if parent_mean is None:
            raise NotImplementedError()
        else:
            m = np.abs(parent_mean[0] - parent_mean[1])
            k = k_11 + k_22 - 2*k_12

            a_1 = (-k/lengthscale)+m
            a_2 = (k/lengthscale)+m

            def component(a, neg=False):
                k_sqrt = np.sqrt(2*np.clip(k, settings.jitter))
                #k_sqrt = np.sqrt(2*k)
                
                b = (- a)/k_sqrt

                c_1 = variance 
                c_2 = -(1/k_sqrt)*(m**2 - a**2)

                if neg is False:
                    return  c_1* 0.5*(1+ erf(b))*np.exp(c_2)
                else: 
                    return  c_1* 0.5*( 1- erf(b))*np.exp(c_2)

            c_1 = component(a_1, neg=True)
            c_2 = component(a_2, neg=False)

            Kij = c_1 + c_2

        return Kij

        #return variance * (1/np.sqrt(1 + (k_11 + k_22 - 2*k_12)/(lengthscale)))

class DeepSMComponent(DeepStationary):
    def __init__(
        self, 
        kernel: Optional['Kernel'] = None,
        parent_model: Optional['Model'] = None,
        lengthscale: Optional[np.ndarray] = None, 
        variance: Optional[np.ndarray] = None, 
        input_dim: Optional[int] = 1, 
        active_dims: Optional[np.ndarray] = None
    ):
        super(DeepSMComponent, self).__init__(lengthscale, variance, input_dim, active_dims)

        self.parent_kernel = kernel
        self.parent_model = parent_model

    @batch
    def lengthscales(self, raw_getter) -> np.ndarray:
        """ \mu is not constrained to be positive in the SM kernel. """
        return raw_getter()

    def _K_scaler(self, x1, x2, variance, lengthscale):
        #TODO: generalise to multi dimensions

        parent_mean, parent_k = self.propogate_parent(x1, x2)

        chex.assert_rank(parent_k, 2)

        k_11 = parent_k[0, 0]
        k_12 = parent_k[0, 1]
        k_22 = parent_k[1, 1]
        
        if parent_mean is None:
            raise NotImplementedError()
        else:
            m = np.abs(parent_mean[0] - parent_mean[1])
            k = k_11 + k_22 - 2*k_12
            k = np.clip(k, settings.jitter)

            v = variance
            mu = lengthscale

            lam = 1/(4*np.pi*np.pi*v)

            b = (1/((1/k)+(1/lam)))
            a = (1/b) * ((1/k)*m)

            c = np.sqrt(2*np.pi*variance)

            gauss = lambda x, m, v : (1/np.sqrt(2*np.pi*v))*np.exp(-(1/(2*v))*(x-m)**2)

            print('exp(): ', np.exp((-1/4)*(2*np.pi*mu)**2))
            print('np.sqrt(np.pi/(2*b)): ', np.sqrt(np.pi/(2*b)))

            Kij = np.sqrt(2*np.pi*lam)*gauss(m, 0, k+lam) * np.exp((-1/2)*(b*(2*np.pi*mu))**2)
            
            #Kij = np.sqrt(2*np.pi*lam)*np.cos(2*np.pi*mu*a)*np.exp(-(1/(2*b))*(2*np.pi*mu)**2)*gauss(m, 0, k+lam)

        return Kij

