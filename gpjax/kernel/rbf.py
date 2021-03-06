from . import StationaryKernel
import jax
import jax.numpy as np
import chex

class RBF(StationaryKernel):
    def _K_scaler(self, x1, x2, variance, lengthscale):
        #ensure scalar inputs
        chex.assert_rank(x1, 0)
        chex.assert_rank(x2, 0)
        chex.assert_rank(variance, 0)
        chex.assert_rank(lengthscale, 0)

        return variance * np.exp(-((x1-x2)**2)/lengthscale)

    
