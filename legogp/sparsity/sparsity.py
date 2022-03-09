import objax
import jax.numpy as np

from ..parameter import Parameter

class Sparsity(objax.Module):
    pass

class FreeSparsity(Sparsity):
    pass

class StructuredSparsity(Sparsity):
    pass


class NoSparsity(FreeSparsity):
    def __init__(self, Z: np.ndarray):
        super(NoSparsity, self).__init__()
        self.raw_Z = Parameter(np.array(Z), constraint=None, train=False, name='Z')

    @property
    def Z(self):
        return self.raw_Z.value

class FullSparsity(FreeSparsity):
    def __init__(self, Z: np.ndarray):
        super(FullSparsity, self).__init__()
        self.raw_Z = Parameter(np.array(Z), constraint=None, name='Z')

    @property
    def Z(self):
        return self.raw_Z.value


class SpatialSparsity(StructuredSparsity):
    def __init__(self, Z: np.ndarray):
        super(FullSparsity, self).__init__()
        self.raw_Z = Parameter(np.array(Z), constraint=None, name='SpatialZ')

    @property
    def Z(self):
        return self.raw_Z.value


