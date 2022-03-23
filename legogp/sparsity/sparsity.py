"""
Sparsity excepts with a numpy array or a parameter object.
"""
import objax
import jax.numpy as np

from ..data import Input
from ..parameter import Parameter

class Sparsity(Input):
    @property
    def X(self):
        return self.Z

class FreeSparsity(Sparsity):
    pass

class StructuredSparsity(Sparsity):
    pass

class NoSparsity(FreeSparsity):
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
    def __init__(self, Z: np.ndarray = None, Z_ref: Parameter = None):

        if Z is not None:
            self.raw_Z = Parameter(np.array(Z), constraint=None, name='SpatialZ')
        else:
            self.raw_Z = Z_ref

    @property
    def Z(self):
        return self.raw_Z.value


