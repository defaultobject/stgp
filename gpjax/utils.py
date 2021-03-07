import jax
import jax.numpy as np


def ensure_array(a):
    return np.array(a)


def ensure_float(a):
    return float(a)

def key_that_ends_with(d: dict, k: str):
    for key in d.keys():
        if key.endswith(k):
            return key
    return None
