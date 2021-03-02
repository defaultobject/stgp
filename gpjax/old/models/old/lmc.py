from . import Model

from ..distributions import (
    GaussianDistribution,
    ZeroMeanGaussianDistribution,
    KernelGaussianDistribution,
    DiagionalGaussianDistribution,
    SparseGaussianDistribution,
    WhitenedGaussianDistribution,
)
from ..likelihoods import LMC_Likelihood, LMC_Constrained_Likelihood
from ..kernels import RBF
from ..decorators import return_gradients
from .. import Parameter
from ..settings import Settings

import jax
import jax.numpy as np
import numpy as onp
from jax import value_and_grad

from jax.config import config

config.update("jax_enable_x64", True)

import json

import matplotlib.pyplot as plt


class LMC(Model):
    def __init__(self, X, Y, likelihood=None, init=None, num_latents=None):
        """
        @TODO: default behaviours:
            if Y in N x P then have P independent latent functions
            if have num_latents and P = 1 then just have P versions of the same GP
            if have num_latents != P && P > 1 then throw run time error
        """
        if init == None:
            init = {}

        self.X = X
        self.Y = Y
        self.Z = X

        if num_latents is None:
            self.num_latents = 1
            self.num_outputs = 1
            self.X = [self.X]
            self.Y = [self.Y]
            self.Z = [self.Z]
        else:
            self.num_latents = num_latents
            self.num_outputs = len(self.Y)

        self.kernel_arr = []
        self.prior_arr = []
        self.approx_posterior_arr = []

        N = init["N"]

        self.likelihood = None
        if likelihood is None:
            self.likelihood = LMC_Likelihood(
                num_outputs=self.num_outputs, num_latents=self.num_latents
            )

        for latent in range(self.num_latents):
            kernel = RBF(
                params={
                    "lengthscale": np.log(1.0),
                    "variance": np.log(1.0),
                },
                name="RBF_{i}".format(i=latent),
                train=True,
            )

            self.kernel_arr.append(kernel)

            prior = KernelGaussianDistribution(
                init={"kernel": kernel},
                name="KernelGaussianDistribtuion_{i}".format(i=latent),
                train=True,
            )

            self.prior_arr.append(prior)

            approx_posterior = WhitenedGaussianDistribution(
                init={
                    "mean": np.zeros(N)[:, None],
                    "covariance_chol": [init["covariance_chol"], init["N"]],
                    "kernel": kernel,
                    "Z": self.Z[latent],
                },
                name="KernelGaussianDistribtuion_{i}".format(i=latent),
                train=True,
            )
            self.approx_posterior_arr.append(approx_posterior)

    @return_gradients
    def get_objective(self):
        elbo = 0.0

        ell = self.likelihood.expected_log_likelihood(
            self.X, self.Y, self.approx_posterior_arr, return_grad=False
        )

        for latent in range(self.num_latents):
            kl_q = self.approx_posterior_arr[latent].KL(
                self.prior_arr[latent], X=self.Z[latent], return_grad=False
            )
            print("kl_q: ", latent, " - ", kl_q)
            elbo += kl_q

        elbo = ell - elbo
        return -elbo

    def predict(self, XS):
        return self.likelihood.predict(XS, self.approx_posterior_arr)

    def anchor(self, params):
        Parameter.PARAM_DICT = params
        Parameter.update(params)

    def print(self):
        for key, val in Parameter.PARAM_DICT.items():
            print(key, ": transformed: ", Parameter.OBJ_DICT[key]())
