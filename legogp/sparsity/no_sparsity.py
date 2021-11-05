from . import Sparsity
import jax.numpy as np

class NoSparsity(Sparsity):
    def __init__(self, Z: np.ndarray):
        self.raw_Z = Z

    @property
    def Z(self):
        return self.raw_Z

    @property
    def X(self):
        return self.raw_Z
