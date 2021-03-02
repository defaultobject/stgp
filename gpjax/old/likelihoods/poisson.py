from . import Likelihood
from ..parameter import Parameter
from ..distributions import GaussianDistribution
from ..computation import *
from ..computation.gaussian import log_gaussian_scalar
from ..computation.general import (
    inv_positive_transform,
    inv_probit,
    inv_logit,
    log_poisson,
)
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


class PoissonLikelihood(Likelihood):
    def __init__(
        self,
        binsize,
        link="exp",
        name: Optional[str] = None,
        meta: Optional[dict] = None,
        trainable: Optional[bool] = True,
    ) -> None:
        self.meta = meta

        self.name = name
        if self.name is None:
            self.name = "PoissonLikelihood"

        if link == "exp":
            self.link_fn = np.exp
        else:
            raise RuntimeError()

        self.link = link
        self.binsize = binsize

        super(PoissonLikelihood, self).__init__(self.name, self.meta, trainable)

    def log_likelihood(self, data: "Data", F_mu):
        Y = data.Y[0]

        mask = data.mask[0]
        bool_mask = np.logical_not(mask)

        F_mu = F_mu[0]

        Y = np.reshape(Y, [-1, 1])
        F_mu = np.reshape(F_mu, [-1, 1])

        Y = Y[bool_mask, :]
        F_mu = F_mu[bool_mask, :]

        ll = log_poisson(Y, self.link_fn(F_mu) * self.binsize)

        return ll

    def eval(self, F):
        raise NotImplementedError()

    def conditional_var(self, F_mu):
        return self.conditional_mean(F_mu)

    def conditional_mean(self, F_mu: List[np.ndarray]):
        return self.link_fn(F_mu[0]) * self.binsize

    def propogate_moments(self, F_m: List[np.ndarray], F_mu_for_var: List[np.ndarray]):
        raise NotImplementedError()
