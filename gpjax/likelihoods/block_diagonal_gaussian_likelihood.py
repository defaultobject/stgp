from . import Likelihood
from ..parameter import Parameter
from ..distributions import GaussianDistribution
from ..computation import *
from ..computation.gaussian import log_gaussian_diagonal
from ..computation.general import inv_positive_transform
from ..settings import Settings
import warnings

from ..decorators import return_gradients

import gpjax
import jax.numpy as np
from jax.experimental import loops
from jax.scipy.special import erfc
from jax.nn import softplus
from jax import jit, partial

import typing
from typing import Optional, List, Union



class BlockDiagonalGaussianLikelihood(Likelihood):
    def __init__(self, variances:Optional[np.ndarray]=None, name: Optional[str]=None,  meta: Optional[dict]=None, trainable: Optional[bool]=True, constraint='block lower triangular') -> None:
        self.meta = meta

        if variances is None:
            if Settings.strict_mode:
                raise RuntimeError('GaussianLikelihood variance is not initalised')

            warnings.warn('GaussianLikelihood variance is not initalised. Default will be used.')
            raise NotImplementedError()

        self.name=name
        if self.name is None:
            self.name = 'BlockDiagonalGaussianLikelihood'

        super(BlockDiagonalGaussianLikelihood, self).__init__(self.name, self.meta, trainable)

        self.constraint = constraint
        #Set up likelihood jax parameters
        scope = 'hyperparameter'
        #self.variances = variances
        #self.variances = self.parameter(val=variances, constraint=constraint, scope=scope, train=trainable, module_name=self.name, param_name='variance')
        self.variances = self.parameter(val=variances, scope=scope, meta={'N': self.meta['N'][1]}, constraint='block lower triangular', train=trainable, module_name=self.name, param_name='covariance_chol')

    @property
    def variance(self):
        covar_chol = self.variances.val

        def block_wise_cholesky_multiply(var_chol):
            return var_chol @ var_chol.T

        S = jax.vmap(block_wise_cholesky_multiply, in_axes=(0), out_axes=0)(covar_chol)

        return S
        #return self.variances.val

    def log_likelihood(self, Y, F_mu):
        raise NotImplementedError()
        #return log_gaussian_diagonal(Y,  F_mu, self.variance)

    def get_variance_at_n(self, n, d):
        return self.variance[n]





