import jax.numpy as jnp

from jax.config import config

config.update("jax_enable_x64", True)
from jax import jit, partial


@jit
def gaussian_expected_log_likelihood(Y, noise, q_mu, q_covar_diag):
    N = Y.shape[0]
    c1 = -0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(noise)

    err = Y - q_mu
    err = jnp.sum(jnp.matmul(err.T, err))

    return N * c1 - 0.5 * (err + jnp.sum(q_covar_diag)) / noise


@jit
def gaussian_log_likelihood(Y, noise, f):
    N = Y.shape[0]
    c1 = -0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(noise)

    err = Y - f
    err = jnp.sum(jnp.matmul(err.T, err))

    return N * c1 - 0.5 * (err) / noise
