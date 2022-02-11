import objax
import jax.numpy as np

class Sparsity(objax.Module):
    @property
    def Z(self):
        raise NotImplementedError()

class NoSparsity(Sparsity):
    def __init__(self, Z: np.ndarray):
        super(NoSparsity, self).__init__()
        self.raw_Z = objax.StateVar(np.array(Z))

    @property
    def Z(self):
        return self.raw_Z.value

class FullSparsity(Sparsity):
    def __init__(self, Z: np.ndarray):
        super(FullSparsity, self).__init__()
        self.raw_Z = objax.TrainVar(np.array(Z))

    @property
    def Z(self):
        return self.raw_Z.value


