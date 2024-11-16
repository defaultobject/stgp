from .kernel import Kernel
import jax.numpy as np

from . import  MarkovKernel
from .. import Parameter
from ..utils.utils import ensure_array, ensure_float




class BiasKernel(Kernel):
    def K_diag(self, X1):
        N = X1.shape[0]
        return np.ones(N)

    def _K(self, X1, X2):
        N1 = X1.shape[0]
        N2 = X2.shape[0]

        return np.ones([N1, N2])

class ConstantKernel(MarkovKernel):
    def __init__(
        self,
        variance = None,
    ) -> None:

        super(ConstantKernel, self).__init__(1, None)

        if variance is None:
            variance = 1.0
        else:
            variance = ensure_float(variance)

        variance = ensure_array(variance)
        self.variance_param = Parameter(variance, constraint='positive', name='ConstantKernel/Variance')

    def K_diag(self, X1):
        N = X1.shape[0]
        return self.variance*np.ones(N)

    def _K_scaler(self, x1, x2):
        return self.variance

    def fix(self):
        self.variance_param.fix()

    def release(self):
        self.variance_param.release()

    @property
    def variance(self) -> np.ndarray:
        return self.variance_param.value

    def state_size(self):
        return 1

    def state_space_dim(self):
        return 1

    def to_ss(self, X_spatial=None):
        F = np.array([[0.0]])
        L = np.array([[1.0]])
        Qc = 0.0
        H = np.array([[1.0]])
        minf = np.array([[0.0]])
        Pinf = self.variance
        return F, L, Qc, H, minf, Pinf

    def expm(self, dt, X_spatial=None):
        return np.exp(0.0)

