import jax.numpy as np
from jax import jit
from functools import partial

@partial(jit, static_argnums=(0, 1))
def data_order_to_output_order(num_outputs: int, N: int):
    total_N = N*num_outputs

    i = np.hstack([np.arange(i , total_N, N) for i in range(N)])
    permutation = np.eye(total_N)[i]

    return permutation
