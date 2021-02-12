from . import Likelihood
from ..parameter import Parameter
from ..distributions import GaussianDistribution
from ..computation import *
from ..computation.gaussian import log_gaussian_scalar
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


class GaussianLikelihood(Likelihood):
    def __init__(self, variances:Optional[np.ndarray]=None, name: Optional[str]=None,  meta: Optional[dict]=None, trainable: Optional[bool]=True) -> None:
        self.meta = meta

        if variances is None:
            if Settings.strict_mode:
                raise RuntimeError('GaussianLikelihood variance is not initalised')

            warnings.warn('GaussianLikelihood variance is not initalised. Default will be used.')
            variances = inv_positive_transform(1.0)

        self.name=name
        if self.name is None:
            self.name = 'GaussianLikelihood'

        super(GaussianLikelihood, self).__init__(self.name, self.meta, trainable)

        #Set up likelihood jax parameters
        scope = 'hyperparameter'
        self.variances = self.parameter(val=variances, constraint='positive', scope=scope, train=trainable, module_name=self.name, param_name='variance')

    @property
    def variance(self):
        return self.variances.val

    def get_variance_at_n(self, n, d):
        return self.variance * np.eye(d)

    def log_likelihood(self, Y, F_mu):
        return log_gaussian_scalar(Y,  F_mu, self.variance)

    def predict_mean_var(self, mean, var):
        return mean, var+self.variance



