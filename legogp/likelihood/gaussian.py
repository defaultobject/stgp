"""Gaussian likelihood."""
import objax
import jax
import jax.numpy as np
from . import Likelihood, FullLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood

from .. import Parameter

from ..computation.parameter_transforms import inv_positive_transform, positive_transform
from ..computation.gaussian import log_gaussian_scalar
from ..computation.matrix_ops import vectorized_lower_triangular_cholesky, vectorized_lower_triangular

class FullGaussian(FullLikelihood):
    def __init__(self, dim: int = None, variance=None, train=True):
        #if (block_size is None and num_blocks is None) or variance is None:
        #    raise NotImplementedError()

        self.dim = dim

        if variance is None:
            variance = np.eye(dim)

        # TODO: ensure positivity here
        self.variance_param = Parameter(
            variance, 
            constraint=None, 
            name ='FullGaussian/variance', 
            train=True
        )

class BlockDiagonalGaussian(BlockDiagonalLikelihood):
    def __init__(self, block_size:int=None, num_blocks:int=None, variance=None, train=True):
        #if (block_size is None and num_blocks is None) or variance is None:
        #    raise NotImplementedError()

        self.block_size = block_size
        self.num_blocks = num_blocks

        if variance is None:
            variance = np.tile(np.eye(block_size), [num_blocks, 1, 1])

        chol = vectorized_lower_triangular_cholesky(variance)

        self.variance_param = Parameter(
            variance, 
            inv_constraint_fn = vectorized_lower_triangular_cholesky, 
            constraint_fn = lambda x: vectorized_lower_triangular(x, self.block_size), 
            name ='BlockGaussian/variance', 
            train=True
        )

    @property
    def variance(self) -> np.ndarray:
        var_chol =  self.variance_param.value
        # Compute LL^T for each block
        return var_chol @ np.transpose(var_chol, [0, 2, 1])

    @property
    def full_variance(self) -> np.ndarray:
        return jax.scipy.linalg.block_diag(*self.variance)

class DiagonalGaussian(DiagonalLikelihood):
    """Gaussian likelihood."""

    def __init__(self, variance=None, train=True):

        if variance is None:
            raise NotImplementedError()

        self.variance_param = Parameter(
            variance, 
            constraint='positive', 
            name ='Gaussian/variance', 
            train=True
        )

    @property
    def variance(self) -> np.ndarray:
        return np.squeeze(self.variance_param.value)

    def conditional_var(self, f):
        return self.variance

    def conditional_mean(self, f):
        return f

class Gaussian(DiagonalGaussian):
    """Gaussian likelihood."""

    def __init__(self, variance=None):

        if variance is None:
            variance = 1.0

        self.variance_param = Parameter(
            variance, 
            constraint='positive', 
            name ='Gaussian/variance', 
            train=True
        )

    @property
    def variance(self) -> np.ndarray:
        return self.variance_param.value

    def log_likelihood_scalar(self, y, f):
        ll = log_gaussian_scalar(y, f, self.variance)
        return ll

    def conditional_var(self, f):
        return self.variance

    def conditional_mean(self, f):
        return f

class GaussianParameterised(Likelihood):
    """Gaussian likelihood."""

    def __init__(self, kernel):
        self.kernel = kernel

    def variance(self, X) -> np.ndarray:
        return self.kernel.K(X, X)
