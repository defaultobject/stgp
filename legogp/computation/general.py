import jax
import jax.numpy as np
from jax import jit
from jax.scipy.special import erf, gammaln

@jit
def log_poisson(x, lam):
    return x * np.log(lam) - lam - gammaln(x + 1.0)
