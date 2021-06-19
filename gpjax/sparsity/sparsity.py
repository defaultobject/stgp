import objax
import jax.numpy as np
from ..batching import batch

class Sparsity(objax.Module):
    def __init__(self, Z: np.ndarray):
        self.raw_Z = objax.TrainVar(Z)
        #self.raw_Z = objax.StateVar(Z)

    @batch
    def Z(self, raw_fn):
        return raw_fn()

