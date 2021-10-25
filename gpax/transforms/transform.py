"""Base transform class."""

import objax
from .. import Node


class Transform(objax.Module, Node):
    """All transforms must be define a forward or inverse method."""
    def __init__(self):
        self._num_latents = None
        self._num_outputs = None

    def forward(self):
        """Compute f=T(x)."""
        pass

    def inverse(self):
        """Compute x=T^{-1}(f)."""
        pass

    @property
    def num_latents(self):
        return self._num_latents

    @property
    def num_outputs(self):
        return self._num_latents

    def get_batches(self):
        return self.batches

class LinearTransform(Transform):
    pass

class NonLinearTransform(Transform):
    pass


class ElementWiseTransform(Transform):
    def __init__(self):
        super(ElementWiseTransform, self).__init__()
        self._num_latents = 1
        self._num_outputs = 1
