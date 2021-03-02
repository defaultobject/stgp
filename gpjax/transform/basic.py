"""Single input and outputs transforms."""
from .transform import Transform

import jax.numpy as np


class Exp(Transform):
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


class Affine(Transform):
    """Affine Function."""


class Boxcox(Transform):
    """Boxcox Function."""


class Sinh_Arcsinh(Transform):
    """Sinh_Arcsinh Function."""


class Tanh(Transform):
    """Sinh_Arcsinh Function."""
