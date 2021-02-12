from .. import Parameter
from .. import Module
from ..computation.general import inv_positive_transform, cholesky, cholesky_solve
from ..settings import Settings
import warnings

import jax
import jax.numpy as np
from jax import jit, partial


from abc import ABC
from abc import abstractmethod

import scipy
from scipy.special import binom
from scipy import math
from math import factorial

import typing
from typing import List, Optional, Union

from jax.numpy import vectorize


class Kernel(Module):
    def __init__(self, lengthscales: Optional[np.ndarray]=None, variances: Optional[np.ndarray]=None, name: Optional[str]=None, meta: Optional[dict]=None, trainable: Optional[bool]=True, input_dim: Optional[int]=1, sparsity=None) -> None:
        super(Kernel, self).__init__(name)

        if sparsity is None:
            sparsity = 'none'

        self.sparsity=sparsity
        self.name = name
        self.meta = meta
        self.trainable=trainable
        self.input_dim=input_dim

        if lengthscales is None:
            if Settings.strict_mode:
                raise RuntimeError('Kernel Lengthscale is not initalised')

            warnings.warn('Kernel lengthscale is not initalised. Default will be used.')

            lengthscales = np.array([inv_positive_transform(1.0) for i in range(self.input_dim)])

        else:
            if type(lengthscales) is not np.ndarray:
                lengthscales == np.array(lengthscales)

        if variances is None:
            if Settings.strict_mode:
                raise RuntimeError('Kernel variance is not initalised')

            warnings.warn('Kernel variance is not initalised. Default will be used.')

            variances = np.array([inv_positive_transform(1.0) for i in range(self.input_dim)])
        else:
            if type(variances) is not np.ndarray:
                variances == np.array(variances)

        scope = 'hyperparameter'
        self.lengthscales = self.parameter(val=lengthscales, constraint='positive', train=trainable, scope=scope, module_name=self.name, param_name='lengthscales')
        self.variances = self.parameter(val=variances, constraint='positive', train=trainable, scope=scope, module_name=self.name, param_name='variances')

    @property
    def lengthscale(self) -> np.ndarray:
        return self.lengthscales.val

    @property
    def variance(self) -> np.ndarray:
        return self.variances.val

    def _get_input(self, X1, X2, active_dims):
        if X2 is None:
            X2 = X1
        return X1, X2

    @abstractmethod 
    def cf_to_ss(self):
        pass

    @abstractmethod 
    def expm(self, dt):
        pass

    #TODO: figure out how to jit this function
    #@partial(jit, static_argnums=(3))
    def K(self, X1, X2, active_dims: Optional[list]=None):   

        #return self._K(X1, X2, 0)
        if active_dims is None:
            active_dims = np.array(list(range(self.input_dim)))
        #automatically get product of last dimenion
        #and support arbitrary batching of kernels

        #remove all dimensions not in the active dims
        #X1 = X1[:, active_dims]
        #X2 = X2[:, active_dims]

        def get_k(X1, X2):
            #adds on an additional dimension to the input so that batching and be done over the last dimension D
            #returns product (element wise/hardamard) over the axis D

            X1 = np.expand_dims(X1, 1)
            X2 = np.expand_dims(X2, 1)

            input_dims =  np.array(list(range(self.input_dim)))[None, :] #1 x D

            k_xx = jax.vmap(self._K, (2, 2, 1), 0)(X1, X2, input_dims)
            #remove all dimensions not in the active dims
            #TODO: not the most efficient because every dimension must be evaluataed
            k_xx = k_xx[active_dims, ...]

            k_xx = np.prod(k_xx, axis=0)
            return k_xx

        return vectorize(get_k, signature='(a,c),(b,c)->(a,b)')(X1, X2)
        
    def K_diag(self, X1, active_dims: Optional[list]=None):
        if active_dims is None:
            active_dims = np.array(list(range(self.input_dim)))

        return self._K_diag(X1, active_dims)

class RBF(Kernel):
    def __init__(self, lengthscales: Optional[np.ndarray]=None, variances: Optional[np.ndarray]=None, name: Optional[str]=None, meta: Optional[dict]=None, trainable: Optional[bool]=True, input_dim: Optional[int]=1):

        if name is None:
            name = 'RBF'

        super(RBF, self).__init__(lengthscales, variances, name, meta, trainable, input_dim)

    def cf_to_ss(self):
        raise NotImplementedError('RBF kernel has not been implemented for state space use')

    def expm(self, dt):
        raise NotImplementedError('RBF kernel has not been implemented for state space use')

    #@partial(jit, static_argnums=(3))  
    def _K(self, X1: np.ndarray, X2: np.ndarray, i: np.ndarray) -> np.ndarray:
        diff = X1-X2.T
        return self.variance[i] * np.exp(-0.5*(diff**2)/(self.lengthscale[i]**2))

    #@partial(jit, static_argnums=(2))  
    def _K_diag(self, X1, active_dims):
        return np.prod(self.variance[active_dims])*np.ones(X1.shape[0])

class Matern32(Kernel):
    def __init__(self, lengthscales: Optional[np.ndarray]=None, variances: Optional[np.ndarray]=None, name: Optional[str]=None, meta: Optional[dict]=None, trainable: Optional[bool]=True, input_dim: Optional[int]=1, X_space: Optional[np.ndarray]=None):

        if name is None:
            name = 'Matern32'

        self.X_space=X_space

        super(Matern32, self).__init__(lengthscales, variances, name, meta, trainable, input_dim)

    def set_spatial_locations(self, X_space, sparsity='none'):
        """
            X_space: [0, x_space] where the time dimension has been padded with zeros
        """
        self.sparsity=sparsity
        self.X_space = X_space

    def cf_to_ss_temporal(self):
        #temporal so input dim in 1
        v = 3.0/2.0
        D = int(v+0.5)


        lam = (3.0 ** 0.5) / self.lengthscale[0]
        F = np.array([[0.0,       1.0],
                      [-lam ** 2, -2 * lam]])

        L = np.array([
            [0.0],
            [1.0]
        ])

        #measurement model matrix
        H = np.array([[1.0, 0.0]])

        Qc = np.array(12.0 * 3.0 ** 0.5 / self.lengthscale[0] ** 3.0 * self.variance[0])

        Pinf = np.array([[self.variance[0], 0.0],
                         [0.0, 3.0 * self.variance[0] / self.lengthscale[0] ** 2.0]])
        
        return F, L, Qc, H, Pinf

    def get_H(self, r):
        _, _, _, H_temporal, _ = self.cf_to_ss_temporal()

        X_space = sparsity.Z
        num_spatial= X_space.shape[0]

        eye = np.eye(num_spatial)

        self.sparsity = 'dtc'
        if True or type(self.sparsity) == NoSparsity():
            A = eye

        elif self.sparsity == 'dtc':
            print(r.shape)
            print(self.X_space.shape)
            Kmm = self.K(self.X_space, self.X_space, active_dims=list(range(1, self.input_dim)))
            Knm = self.K(r, self.X_space, active_dims=list(range(1, self.input_dim)))

            Kmm_chol = cholesky(Kmm+Settings.jitter*eye)

            A = cholesky_solve(Kmm_chol, Knm.T).T

        H = np.kron(A, H_temporal)
        return H

    def cf_to_ss_spatial(self, sparsity):
        X_space = sparsity.Z

        num_spatial= X_space.shape[0]

        eye = np.eye(num_spatial)

        K_spatial = self.K(X_space, X_space, active_dims=list(range(1, self.input_dim)))
        F_temporal, L_temporal, Qc_temporal, H_temporal, P_inf_temporal = self.cf_to_ss_temporal()


        F = np.kron(eye, F_temporal)
        L = np.kron(eye, L_temporal)

        #TODO generalise? #not needed atm
        H = np.kron(eye, H_temporal)

        Qc = np.kron(K_spatial, Qc_temporal)
        Pinf = np.kron(K_spatial, P_inf_temporal)

        if False:
            print('X_space ', X_space.shape)
            print('K_spatial: ', K_spatial.shape, K_spatial)
            print('F: ', F.shape, F)
            print('L: ', L.shape, L)
            print('H: ', H.shape, H)
            print('Qc: ', Qc.shape, Qc)
            print('Pinf: ', Pinf.shape, Pinf)

        return F, L, Qc, H, Pinf

    #@jit
    def cf_to_ss(self, sparsity):
        if self.input_dim == 1:
            return self.cf_to_ss_temporal()
        else:
            return self.cf_to_ss_spatial(sparsity)

    #@partial(jit, static_argnums=(1))  
    def expm(self, dt, sparsity):
        """closed form matrix exponential A = expm(F * dt)"""
        lam = np.sqrt(3.0) / self.lengthscale[0]
        A = np.exp(-dt * lam) * (dt * np.array([[lam, 1.0], [-lam**2.0, -lam]]) + np.eye(2))

        if self.input_dim != 1:
            X_space = sparsity.Z
            #spatio-temporal setting
            #because Fdt is block diagional exp(Fdt) is just a block diagional matrix

            num_spatial= X_space.shape[0]

            eye = np.eye(num_spatial)
            A = np.kron(eye, A)

        return A

    #@partial(jit, static_argnums=(3))  
    def _K(self, X1, X2, i):
        """
                K(X1, X2) = σ² (1 + √3 (X1-X2)/l) exp{-√3 (X1-X2)/l}
        """

        diff = X1-X2.T
        r2 = np.square(diff / self.lengthscale[i])
        r = np.sqrt(np.clip(r2, 1e-36))

        sqrt3 = np.sqrt(3.0)

        return self.variance[i] * (1.0 + sqrt3 * r) * np.exp(-sqrt3 * r)

    #@partial(jit, static_argnums=(2))  
    def _K_diag(self, X1, active_dims):
        return np.prod(self.variance[active_dims])*np.ones(X1.shape[0])
