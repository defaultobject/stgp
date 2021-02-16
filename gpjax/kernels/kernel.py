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
from ..computation.general import inv_positive_transform, positive_transform
from ..utils import ensure_array, ensure_float

class Kernel(objax.Module):
    def __init__(self):
        pass

    #kernel combinations
    def __add__(self, kern_2): 
        return SumKernel(self, kern_2)

    def __mul__(self, kern_2): 
        return ProductKernel(self, kern_2)

    @abstractmethod
    def K(self, X1: np.array, X2: np.array):
        raise NotImplementedError()

    @abstractmethod
    def K_diag(self, X: np.array):
        raise NotImplementedError()

class CombinationKernel(Kernel):
    def __init__(self, k1: 'Kernel', k2: 'Kernel'):
        self.k1 = k1
        self.k2 = k2

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


class MarkovKernel(Kernel):
    pass


class StationaryKernel(Kernel):
    def __init__(
        self, 
        lengthscales: Optional[np.ndarray]=None, 
        variance: Optional[float]=None, 
        input_dim: Optional[int]=1, 
        active_dim: Optional[np.ndarray]=None,
    ) -> None:

        super(Kernel, self).__init__()

        chex.assert_type(input_dim, int)

        #input admin
        if lengthscales is None:
            lengthscales = np.array([1.0]*input_dim)
        else:
            lengthscales = ensure_array(lengthscales)

        if variance is None:
            variance = 1.0
        else:
            ensure_float(variance)

        chex.assert_shape(lengthscales, [input_dim])
        chex.assert_rank(variance, 0) #scalar

        #register lengthscales and variances
        self.raw_lengthscale = objax.StateVar(inv_positive_transform(lengthscales))
        self.raw_variance = objax.StateVar(inv_positive_transform(variance))

    @property
    def lengthscales(self) -> np.ndarray:
        return positive_transform(self.raw_lengthscale.value)

    @property
    def variances(self) -> np.ndarray:
        return positive_transform(self.raw_variance.value)

class NonStationaryKernel(Kernel):
    def __init__(self) -> None:
        super(Kernel, self).__init__()

        pass

