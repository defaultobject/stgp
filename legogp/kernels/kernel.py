import objax
import chex
import jax
import jax.numpy as np

from jax import jit, partial
from abc import ABC
from abc import abstractmethod
from jax.numpy import vectorize
import typing
from typing import List, Optional, Union
from ..computation.parameter_transforms import inv_positive_transform, positive_transform
from ..utils.utils import ensure_array, ensure_float
from ..batching import batch


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

    @abstractmethod
    def K(self, X1: np.array, X2: np.array):
        chex.assert_rank([X1, X2], [2, 2])
        chex.assert_equal(X1.shape[1], X2.shape[1])

        if self.active_dims is None:
            _X1, _X2 = X1, X2
        else:
            _X1 = X1[:, self.active_dims]
            _X2 = X2[:, self.active_dims]


        return self._K(_X1, _X2)

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
    def K(self, X1: np.array, X2: np.array):
        return self.k1.K(X1, X2) + self.k2.K(X1, X2)

    def K_diag(self, X1: np.array):
        return self.k1.K_diag(X1) + self.k2.K_diag(X1)


class ProductKernel(CombinationKernel):
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

class WhiteNoiseKernel(Kernel):
    def __init__(self, variance: Optional[np.ndarray] = None):
        if variance is None:
            variance = 1.0
        else:
            ensure_float(variance)

        chex.assert_rank(variance, 0)  # scalar
        self.raw_variance = objax.TrainVar(inv_positive_transform(variance))

    @batch
    def variance(self, raw_getter) -> np.ndarray:
        return positive_transform(raw_getter())

    def K_diag(self, X1):
        return self.variance * np.ones(X1.shape[0])

    def K(self, X1, X2):
        # X1 in N1 x D
        # X2 in N2 x D

        k = ((X1-X2.T)==0).astype(float) * self.variance

        return k

class StationaryKernel(Kernel):
    def __init__(
        self,
        lengthscales: Optional[np.ndarray] = None,
        variance: Optional[np.ndarray] = None,
        input_dim: Optional[int] = 1,
        active_dims: Optional[np.ndarray] = None,
    ) -> None:

        super(StationaryKernel, self).__init__(input_dim, active_dims)

        if lengthscales is None:
            lengthscales = np.array([1.0] * input_dim)
        else:
            lengthscales = ensure_array(lengthscales)

        if variance is None:
            variance = 1.0
        else:
            ensure_float(variance)

        chex.assert_shape(lengthscales, [input_dim])

        # register lengthscales and variances
        self.raw_lengthscales = objax.TrainVar(inv_positive_transform(lengthscales))
        self.raw_variance = objax.TrainVar(inv_positive_transform(variance))

    @batch
    def lengthscales(self, raw_getter) -> np.ndarray:
        return positive_transform(raw_getter())

    @batch
    def variance(self, raw_getter) -> np.ndarray:
        return positive_transform(raw_getter())

    def K_diag(self, X1):
        return self.variance * np.ones(X1.shape[0])

    def _K(self, X1, X2):
        D = X1.shape[1]

        def _K_d2(x1, x2):
            #vectorised over 2nd input
            chex.assert_rank(x1, 1)
            chex.assert_rank(x2, 1)

            chex.assert_equal(x1.shape[0], D)
            chex.assert_equal(x2.shape[0], D)

            k_d1_d2 = jax.vmap(self._K_scaler, in_axes=[0, 0, None, 0])(x1, x2, self.variance, self.lengthscales)

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



class NonStationaryKernel(Kernel):
    def __init__(self) -> None:
        super(Kernel, self).__init__()

        pass
