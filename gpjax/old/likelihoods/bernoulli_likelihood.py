from . import Likelihood
from ..parameter import Parameter
from ..distributions import GaussianDistribution
from ..computation import *
from ..computation.gaussian import log_gaussian_scalar
from ..computation.general import inv_positive_transform, inv_probit, inv_logit
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


class BernoulliLikelihood(Likelihood):
    def __init__(
        self,
        link="probit",
        name: Optional[str] = None,
        meta: Optional[dict] = None,
        trainable: Optional[bool] = True,
    ) -> None:
        self.meta = meta

        self.name = name
        if self.name is None:
            self.name = "BernoulliLikelihood"

        if link == "probit":
            self.link_fn = inv_probit
        elif link == "logit":
            self.link_fn = inv_logit
        else:
            raise RuntimeError()

        self.link = link

        super(BernoulliLikelihood, self).__init__(self.name, self.meta, trainable)

    def log_likelihood(self, data: "Data", F_mu):
        Y = data.Y[0]
        F_mu = F_mu[0]

        Y = np.reshape(Y, [-1, 1])
        F_mu = np.reshape(F_mu, [-1, 1])

        ll = np.log(
            np.where(np.equal(Y, 1), self.link_fn(F_mu), 1 - self.link_fn(F_mu))
        )

        return ll

    def eval(self, F):
        return self.link_fn(F)

    def conditional_var(self, F_mu):
        p = self.conditional_mean(F_mu)
        return [p - (p ** 2)]

    def conditional_mean(self, F_mu: List[np.ndarray]):
        return self.link_fn(F_mu[0])

    def propogate_moments(self, F_m: List[np.ndarray], F_mu_for_var: List[np.ndarray]):
        F_mu = F_mu[0]
        F_mu_for_var = F_mu_for_var[0]

        mu_for_mean = self.link_fn(F_mu)

        mu_for_var = self.link_fn(F_mu_for_var)

        return mu_for_mean, mu_for_var ** 2
