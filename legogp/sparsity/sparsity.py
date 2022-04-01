"""
Sparsity accepts a numpy array or a parameter object.
"""
import objax
import jax.numpy as np
from batchjax import batch_or_loop
from ..utils.utils import get_batch_type

from ..data import Input, SpatialTemporalInput
from ..parameter import Parameter

class Sparsity(Input):
    @property
    def X(self):
        return self.Z

class FreeSparsity(Sparsity):
    pass

class StructuredSparsity(Sparsity):
    pass

class NoSparsity(Sparsity):
    def __init__(self, Z: np.ndarray = None, Z_ref: Parameter = None):

        if Z is not None:
            self.raw_Z = Parameter(np.array(Z), constraint=None, train=False, name='Z')
        else:
            self.raw_Z = Z_ref

    @property
    def Z(self):
        return self.raw_Z.value

class FullSparsity(FreeSparsity):
    def __init__(self, Z: np.ndarray = None, Z_ref: Parameter = None):

        if Z is not None:
            self.raw_Z = Parameter(np.array(Z), constraint=None, name='Z')
        else:
            self.raw_Z = Z_ref

    @property
    def Z(self):
        return self.raw_Z.value


class SpatialSparsity(StructuredSparsity):
    def __init__(self, X_time, Z_space):

        self.raw_Z = SpatialTemporalInput(
            X_time = X_time,
            X_space = Z_space,
            train=True
        )

    @property
    def Z(self):
        return self.raw_Z.value

class StackedSparsity(Sparsity):
    def __init__(self, sparsity_arr):
        self.sparsity_arr = objax.ModuleList(sparsity_arr)

    @property
    def Z(self):
        return batch_or_loop(
            lambda Z: Z.Z,
            [self.sparsity_arr],
            [0],
            dim=len(self.sparsity_arr),
            out_dim=1,
            batch_type = get_batch_type(self.sparsity_arr)
        )


class StackedNoSparsity(StackedSparsity, NoSparsity):
    pass
