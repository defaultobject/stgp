from jax.config import config
import jax
import jax.numpy as np
from jax import jit, partial
from jax.ops import index, index_add
from jax.experimental import loops
from jax.scipy.special import erf, gammaln

@jit
def log_poisson(x, lam):
    return x * np.log(lam) - lam - gammaln(x + 1.0)
