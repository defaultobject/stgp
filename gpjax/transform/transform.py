"""Base transform class."""

import objax
from .. import Node


class Transform(objax.Module, Node):
    """All transforms must be define a forward or inverse method."""

    def forward(self):
        """Compute f=T(x)."""
        pass

    def inverse(self):
        """Compute x=T^{-1}(f)."""
        pass

    def number_of_latents(self):
        raise NotImplementedError()

class LinearTransform(Transform):
    pass

class NonLinearTransform(Transform):
    pass


class ElementWiseTransform(Transform):
    pass
