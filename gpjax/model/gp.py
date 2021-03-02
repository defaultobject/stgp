import objax
import jax.numpy as np

from ..decorators import strict_mode_check


class GP(objax.Module):
    @strict_mode_check
    def __init__(self, X: np.ndarray = None , Y: np.ndarray=None , likelihood=None, kernel=None):
        print(X, Y)

    def get_objective():
        pass
