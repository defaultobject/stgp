import jax
import jax.numpy as np
from jax import jit

@jit
def matern32_temporal_expm(dt, lengthscales):
    lam = np.sqrt(3.0) / lengthscales
    A = np.exp(-dt * lam) * (dt * np.array([[lam, 1.0], [-lam**2.0, -lam]]) + np.eye(2))
    return A
