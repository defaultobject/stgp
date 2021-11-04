"""Single input and outputs transforms."""
from .transform import Transform, LinearTransform, ElementWiseTransform

import jax.numpy as np
import objax
from ..utils.utils import ensure_module_list

class Independent(LinearTransform):
    def __init__(self, latents: list):
        self._num_latents = len(latents)
        self._num_outputs = self.num_latents

        self._latents = ensure_module_list(latents)

    @property
    def W(self):
        return np.eye(self.num_latents)

class Identity(ElementWiseTransform):
    pass

class Exp(ElementWiseTransform):
    """Expontial Function."""

    def forward(self, x):
        """Compute f=T(x)."""
        return np.exp(x)

    def inverse(self, f):
        """Compute x=T^{-1}(f)."""
        return np.log(f)


class Log(Exp):
    """Log function. Inverse of Exp function."""

    def forward(self, x):
        """Compute f=T(x)."""
        return super(Log, self).inverse(x)

    def inverse(self, f):
        """Compute x=T^{-1}(f)."""
        return super(Log, self).forward(f)


class Affine(ElementWiseTransform):
    """Affine Function."""


class Boxcox(ElementWiseTransform):
    """Boxcox Function."""


class Sinh_Arcsinh(ElementWiseTransform):
    """Sinh_Arcsinh Function."""


class Tanh(ElementWiseTransform):
    """Sinh_Arcsinh Function."""
