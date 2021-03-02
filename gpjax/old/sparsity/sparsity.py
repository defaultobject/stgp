from ..module import Module

import jax
import jax.numpy as np
import numpy as onp

import typing
from typing import Optional


class Sparsity(Module):
    def __init__(
        self,
        inducing_locations: Optional[np.ndarray] = None,
        trainable: Optional[bool] = True,
        name: Optional[str] = "sparsity",
    ):

        super(Sparsity, self).__init__(name)

        self.name = name

        if inducing_locations is not None:
            self.inducing_locations = self.parameter(
                val=inducing_locations,
                train=trainable,
                module_name=self.name,
                param_name="inducing_locations",
            )

    @property
    def Z(self):
        return self.inducing_locations.val
