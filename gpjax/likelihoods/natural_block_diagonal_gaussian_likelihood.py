from . import Likelihood
from ..parameter import Parameter
from ..distributions import GaussianDistribution
from ..computation import *
from ..computation.gaussian import log_gaussian_diagonal
from ..computation.general import inv_positive_transform
from ..settings import Settings
import warnings

from ..decorators import return_gradients


from ..computation.exponential_family import *

import gpjax
import jax.numpy as np
from jax.experimental import loops
from jax.scipy.special import erfc
from jax.nn import softplus
from jax import jit, partial

import typing
from typing import Optional, List, Union

class NaturalBlockDiagonalGaussianLikelihood(Likelihood):
    def __init__(self, lambda_1:Optional[np.ndarray]=None, covar_chol:Optional[np.ndarray]=None, name: Optional[str]=None,  meta: Optional[dict]=None, trainable: Optional[bool]=True, constraint='positive') -> None:
        self.meta = meta
        self.constraint = constraint

        if lambda_1 is None or covar_chol is None:
            if Settings.strict_mode:
                raise RuntimeError('GaussianLikelihood variance is not initalised')

            warnings.warn('GaussianLikelihood variance is not initalised. Default will be used.')
            raise NotImplementedError()

        self.name=name
        if self.name is None:
            self.name = 'NaturalBlockDiagonalGaussianLikelihood'

        super(NaturalBlockDiagonalGaussianLikelihood, self).__init__(self.name, self.meta, trainable)


        scope = 'variational'
        self.lambda_1 = self.parameter(val=lambda_1, scope=scope, train=trainable, module_name=self.name, param_name='lambda_1')

        self.covar_chol = self.parameter(val=covar_chol, scope=scope, meta={'N': self.meta['N'][1]}, constraint='block lower triangular', train=trainable, module_name=self.name, param_name='covariance_chol')

    @property
    def variance(self):

        covar_chol = self.covar_chol.val

        def block_wise_cholesky_multiply(var_chol):
            return var_chol @ var_chol.T

        S = jax.vmap(block_wise_cholesky_multiply, in_axes=(0), out_axes=0)(covar_chol)

        return S


    def get_variance_at_n(self, n, d):
        return self.variance[n]

    @property
    def Y(self):
        lambda_1 = self.lambda_1.value
        var = self.variance

        def mult(a, b):
            return a@b

        Y = jax.vmap(mult, in_axes=(0, 0), out_axes=0)(var, lambda_1)

        return Y

    def log_likelihood(self, Y, F_mu):
        raise NotImplementedError()
        return log_gaussian_diagonal(Y,  F_mu, self.variance)





