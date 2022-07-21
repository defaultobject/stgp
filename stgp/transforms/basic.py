"""Single input and outputs transforms."""
from .transform import Transform, LinearTransform, ElementWiseTransform, LatentSpecific, ParentPassThrough

import jax.numpy as np
import objax
from ..utils.utils import ensure_module_list
from ..computation.parameter_transforms import softplus, inv_softplus, inv_probit
from ..parameter import Parameter

class InputMeanFunction(LinearTransform, LatentSpecific, ParentPassThrough):
    def __init__(self, latent):
        self._parent = latent

    def mean(self, X):
        return X[:, 0][:, None]


class Identity(LinearTransform):
    def forward(self, x):
        return x

    def inverse(self, f):
        return f

class ReverseFlow(ElementWiseTransform):
    def __init__(self, base_flow):
        self.base_flow = base_flow

    def forward(self, x):
        return self.base_flow.inverse(x)

    def inverse(self, f):
        """Compute x=T^{-1}(f)."""
        return self.base_flow.forward(f)



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

class Softminus(ElementWiseTransform):

    def forward(self, x):
        """Compute f=T(x)."""
        return inv_softplus(x)

    def inverse(self, f):
        """Compute x=T^{-1}(f)."""
        return softplus(f)

class Softplus(ElementWiseTransform):

    def forward(self, x):
        """Compute f=T(x)."""
        return softplus(x)

    def inverse(self, f):
        """Compute x=T^{-1}(f)."""
        return inv_softplus(f)


class Affine(ElementWiseTransform):
    """Affine Function."""
    def __init__(self, a, b, train=True):

        self.a_param = Parameter(
            np.array(a), 
            constraint=None, 
            name ='Affine/a', 
            train=train
        )

        self.b_param = Parameter(
            np.array(b), 
            constraint=None, 
            name ='Affine/b', 
            train=train
        )

    @property
    def a(self):
        return self.a_param.value

    @property
    def b(self):
        return self.b_param.value

    def forward(self, x):
        return x * self.a + self.b

    def inverse(self, f):
        return (f - self.b) / self.a


class Boxcox(ElementWiseTransform):
    """Boxcox Function."""


class Sinh_Arcsinh(ElementWiseTransform):
    """Sinh_Arcsinh Function."""


class Tanh(ElementWiseTransform):
    """Sinh_Arcsinh Function."""

class InvProbit(ElementWiseTransform):
    """Sinh_Arcsinh Function."""
    def forward(self, x):
        return inv_probit(x)
