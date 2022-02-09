import objax
import chex
import jax
import jax.numpy as np

from jax.scipy.linalg import block_diag 
from abc import ABC
from abc import abstractmethod
from jax.numpy import vectorize
import typing
from typing import List, Optional, Union
from ..computation.parameter_transforms import inv_positive_transform, positive_transform
from ..utils.utils import ensure_array, ensure_float
from .. import Parameter


class Kernel(objax.Module):
    def __init__(
        self,
        input_dim: Optional[int] = 1,
        active_dims: Optional[np.ndarray] = None,
    ):
        chex.assert_type(input_dim, int)

        if active_dims is not None:
            active_dims = ensure_array(active_dims)
            chex.assert_rank(active_dims, 1)  # ensure array
            chex.assert_equal(input_dim, active_dims.shape[0])

        self.input_dim = input_dim
        self.active_dims = active_dims

    # kernel combinations
    def __add__(self, kern_2):
        return SumKernel(self, kern_2)

    def __mul__(self, kern_2):
        return ProductKernel(self, kern_2)

    def _apply_active_dim(self, X):
        if self.active_dims is None:
            X = X
        else:
            X = X[:, self.active_dims]

        return X

    @abstractmethod
    def K(self, X1: np.array, X2: np.array):
        chex.assert_rank([X1, X2], [2, 2])
        chex.assert_equal(X1.shape[1], X2.shape[1])

        _X1 = self._apply_active_dim(X1)
        _X2 = self._apply_active_dim(X2)

        return self._K(_X1, _X2)

    def _K(self, X1, X2):
        D = X1.shape[1]
        def _K_d2(x1, x2):
            chex.assert_shape(x1, [D])
            chex.assert_shape(x1, [D])

            k_d1_d2 = jax.vmap(self._K_scaler, in_axes=[0, 0])(x1, x2)
            chex.assert_shape(k_d1_d2, [D])

            k_xx =  np.product(k_d1_d2)
            chex.assert_rank(k_xx, 0)

            return k_xx

        def _K_d1(x1, X2):
            #vectorised over first input
            return jax.vmap(_K_d2, in_axes=[None, 0], out_axes=0)(x1, X2)

        K = jax.vmap(_K_d1, in_axes=[0, None], out_axes=0)(X1, X2)

        chex.assert_equal(K.shape[0], X1.shape[0])
        chex.assert_equal(K.shape[1], X2.shape[0])

        return K

    @abstractmethod
    def K_diag(self, X: np.array):
        raise NotImplementedError()


class CombinationKernel(Kernel):
    def __init__(self, k1: "Kernel", k2: "Kernel"):
        self.k1 = k1
        self.k2 = k2

    def __list__(self):
        if isinstance(self.k1, CombinationKernel):
            k1_arr = list(self.k1)
        else:
            k1_arr = [self.k1]

        if isinstance(self.k2, CombinationKernel):
            k2_arr = list(self.k2)
        else:
            k2_arr = [self.k2]

        return k1_arr + k2_arr

    def __getitem__(self, index):
        """
        When a kernel is defined like:
            k = k1*k2*k3
        The equivalent kernel is:
            ProductKernel(k1, ProductKernel(k2, k3))
        """

        all_kernels = self.__list__()
        return all_kernels[index]


class SumKernel(CombinationKernel):
    def to_ss(self):
        k1_F, k1_L, k1_Qc, k1_H, k1_Pinf = self.k1.to_ss()
        k2_F, k2_L, k2_Qc, k2_H, k2_Pinf = self.k2.to_ss()

        F = block_diag(k1_F, k2_F)
        L = block_diag(k1_L, k2_L)
        Pinf = block_diag(k1_Pinf, k2_Pinf)
        Q = block_diag(k1_Q, k2_Q)
        H = np.hstack(k1_H, k2_H)

        return F, L, Qc, H, Pinf

    def K(self, X1: np.array, X2: np.array):
        return self.k1.K(X1, X2) + self.k2.K(X1, X2)

    def K_diag(self, X1: np.array):
        return self.k1.K_diag(X1) + self.k2.K_diag(X1)


class ProductKernel(CombinationKernel):
    def to_ss(self):
        k1_F, k1_L, k1_Qc, k1_H, k1_Pinf = self.k1.to_ss()
        k2_F, k2_L, k2_Qc, k2_H, k2_Pinf = self.k2.to_ss()

        I_1 = np.eye(k1_F.shape[0])
        I_2 = np.eye(k2_F.shape[0])

        F = np.kron(k1_F, I_1) + np.kron(I_2, k2_F)
        L = np.kron(k1_L, k2_L)
        Q = np.kron(k1_Qc, k2_Qc)
        Pinf = np.kron(k1_Pinf, k2_Pinf)
        H = np.kron(k1_H, k2_H)

        return F, L, Qc, H, Pinf

    def K(self, X1: np.array, X2: np.array):
        return self.k1.K(X1, X2) * self.k2.K(X1, X2)

    def K_diag(self, X1: np.array):
        return self.k1.K_diag(X1) * self.k2.K_diag(X1)

class ConcatationKernel(CombinationKernel):
    def K(self, X1: np.array, X2: np.array):
        return np.array([self.k1.K(X1, X2), self.k2.K(X1, X2)])

    def K_diag(self, X1: np.array):

        return np.array([self.k1.K_diag(X1), self.k2.K_diag(X1)])

class MarkovKernel(Kernel):
    def cf_to_ss_spatial(self, sparsity):
        raise NotImplementedError()

class SpatioTemporalSeperableKernel(MarkovKernel, ProductKernel):
    def __init__(self, K_temporal, K_spatial):
        self.k1 = K_temporal
        self.k2 = K_spatial

    def to_ss(self, X_spatial):
        K_spatial = self.k2.K(X_spatial, X_spatial)

        F, L, Qc, H, Pinf = self.k1.to_ss()

        eye = np.eye(K_spatial.shape[0])

        F_st = np.kron(eye, F)
        L_st = np.kron(eye, L)
        Qc_st = np.kron(K_spatial, Qc)
        H_st = np.kron(eye, H)
        Pinf_st = np.kron(K_spatial, Pinf)

        return F_st, L_st, Qc_st, H_st, Pinf_st

    def state_size(self):
        # only return the temporal state_size 
        return self.k1.state_size()

    def expm(self, dt, X_spatial):
        A_t = self.k1.expm(dt)

        eye = np.eye(X_spatial.shape[0])

        A = np.kron(eye, A_t)

        return A


class WhiteNoiseKernel(Kernel):
    def __init__(
        self,
        input_dim: Optional[int] = 1, 
        active_dims: Optional[np.ndarray] = None
    ):
        super(WhiteNoiseKernel, self).__init__(input_dim, active_dims)

    def _K_scaler(self, x1, x2):
        chex.assert_rank(x1, 0)
        chex.assert_rank(x2, 0)

        return ((x1-x2)==0).astype(float)

class ScaleKernel(Kernel):
    def __init__(
        self,
        kernel: 'Kernel',
        variance: Optional[np.ndarray] = None,
    ) -> None:

        super(ScaleKernel, self).__init__(1, None)
        self.parent_kernel = kernel

        if variance is None:
            variance = 1.0
        else:
            ensure_float(variance)

        self.variance_param = Parameter(variance, constraint='positive')

    @property
    def variance(self) -> np.ndarray:
        return self.variance_param.value

    def K_diag(self, X1):
        return self.variance * self.parent_kernel.K_diag(X1)

    def _K(self, X1, X2):
        return self.variance * self.parent_kernel.K(X1, X2)

class StationaryKernel(Kernel):
    def __init__(
        self,
        lengthscales: Optional[np.ndarray] = None,
        input_dim: Optional[int] = 1,
        active_dims: Optional[np.ndarray] = None,
    ) -> None:

        super(StationaryKernel, self).__init__(input_dim, active_dims)

        if lengthscales is None:
            lengthscales = np.array([1.0] * input_dim)
        else:
            lengthscales = ensure_array(lengthscales)

        chex.assert_shape(lengthscales, [input_dim])

        # register lengthscales and variances
        self.lengthscale_param = Parameter(lengthscales, constraint='positive')

    @property
    def lengthscales(self) -> np.ndarray:
        return self.lengthscale_param.value

    def K_diag(self, X1):
        #TODO: this needs to be multiplied by D, or var is only used once!
        return np.ones(X1.shape[0])

    def _K(self, X1, X2):
        D = X1.shape[1]

        def _K_d2(x1, x2):
            #vectorised over 2nd input
            chex.assert_rank(x1, 1)
            chex.assert_rank(x2, 1)

            chex.assert_equal(x1.shape[0], D)
            chex.assert_equal(x2.shape[0], D)

            k_d1_d2 = jax.vmap(self._K_scaler, in_axes=[0, 0, 0])(x1, x2, self.lengthscales)

            chex.assert_equal(k_d1_d2.shape[0], D)

            k_xx =  np.product(k_d1_d2)

            chex.assert_rank(k_xx, 0)

            return k_xx

        def _K_d1(x1, X2):
            #vectorised over first input
            return jax.vmap(_K_d2, in_axes=[None, 0], out_axes=0)(x1, X2)

        K = jax.vmap(_K_d1, in_axes=[0, None], out_axes=0)(X1, X2)

        chex.assert_equal(K.shape[0], X1.shape[0])
        chex.assert_equal(K.shape[1], X2.shape[0])

        return K

class StationaryVarianceKernel(StationaryKernel):
    def __init__(
        self,
        lengthscales: Optional[np.ndarray] = None,
        variance: Optional[np.ndarray] = None,
        input_dim: Optional[int] = 1,
        active_dims: Optional[np.ndarray] = None,
    ) -> None:

        super(StationaryVarianceKernel, self).__init__(lengthscales, input_dim, active_dims)

        # register lengthscales and variances
        self.variance_param = Parameter(variance, constraint='positive')

    @property
    def variance(self) -> np.ndarray:
        return self.variance_param.value

    def _K_scaler(self, x1, x2, lengthscale):
        return self._K_scaler_with_var(x1, x2, lengthscale, self.variance)

class NonStationaryKernel(Kernel):
    def __init__(self) -> None:
        super(Kernel, self).__init__()

        pass

class Linear(Kernel):
    def _K(self, X1, X2):
        return X1 @ X2.T

    def K_diag(self, X1):
        return np.square(self._apply_active_dim(X1))[:, 0]


