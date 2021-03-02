"""Base transform class."""

import objax


class Transform(objax.Module):
    """All transforms must be define a forward or inverse method."""

    def forward(self):
        """Compute f=T(x)."""
        pass

    def inverse(self):
        """Compute x=T^{-1}(f)."""
        pass
