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


class NaturalDiagonalGaussianLikelihood(Likelihood):
    def __init__(
        self,
        lambda_1: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        meta: Optional[dict] = None,
        trainable: Optional[bool] = True,
        constraint="positive",
    ) -> None:
        self.meta = meta
        self.constraint = constraint

        if lambda_1 is None or covariance is None:
            if Settings.strict_mode:
                raise RuntimeError("GaussianLikelihood variance is not initalised")

            warnings.warn(
                "GaussianLikelihood variance is not initalised. Default will be used."
            )
            raise NotImplementedError()

        self.name = name
        if self.name is None:
            self.name = "NaturalDiagonalGaussianLikelihood"

        super(NaturalDiagonalGaussianLikelihood, self).__init__(
            self.name, self.meta, trainable
        )

        scope = "variational"
        self.lambda_1 = self.parameter(
            val=lambda_1,
            scope=scope,
            train=trainable,
            module_name=self.name,
            param_name="lambda_1",
        )

        self.covariance = self.parameter(
            val=covariance,
            scope=scope,
            constraint="positive",
            train=trainable,
            module_name=self.name,
            param_name="covariance",
        )

    @property
    def variance(self):
        return self.covariance.value

    def get_variance_at_n(self, n, d):
        return self.variance[n]

    @property
    def Y(self):
        lambda_1 = self.lambda_1.value
        variance = self.variance

        return variance * lambda_1

    def log_likelihood(self, Y, F_mu):
        raise NotImplementedError()
        return log_gaussian_diagonal(Y, F_mu, self.variance)
