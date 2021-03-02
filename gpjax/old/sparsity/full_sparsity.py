from . import Sparsity

import jax
import jax.numpy as np

from typing import Optional


class FullSparsity(Sparsity):
    def __init__(
        self,
        inducing_locations: Optional[np.ndarray] = None,
        trainable: Optional[bool] = True,
        name: Optional[str] = "full_sparsity",
    ):
        self.name = name

        super(FullSparsity, self).__init__(inducing_locations, trainable, name)
