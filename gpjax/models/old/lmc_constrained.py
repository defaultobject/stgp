from . import Model, LMC

from ..distributions import GaussianDistribution, ZeroMeanGaussianDistribution, KernelGaussianDistribution, DiagionalGaussianDistribution, SparseGaussianDistribution, WhitenedGaussianDistribution
from ..likelihoods import Likelihood, LMC_Likelihood
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

class LMC_Constrained(LMC):
    def __init__(self, X: np.ndarray, Y: np.ndarray, likelihood: Likelihood=None, init: dict=None, num_latents: int=None, kernels: list=None, priors: list=None, approx_posteriors: list=None):
        """
            @TODO: default behaviours:
                if Y in N x P then have P independent latent functions
                if have num_latents and P = 1 then just have P versions of the same GP
                if have num_latents != P && P > 1 then throw run time error
        """
        if init==None:
            init = {}

        self.X = X
        self.Y = Y
        self.Z = X


        if num_latents is None:
            self.X = [self.X]
            self.Y = [self.Y]
            self.Z = [self.Z]
        else:
            self.num_outputs = len(self.Y)
       
        self.num_latents = self.num_outputs

        self.kernel_arr = []
        self.prior_arr = []
        self.approx_posterior_arr = []

        N = init['N']

        self.likelihood = None
        if likelihood is None:
            self.likelihood = LMC_Constrained_Likelihood(num_outputs = self.num_outputs, num_latents=self.num_latents)

        for latent in range(self.num_latents):
            kernel = RBF(params={
                'lengthscale': np.log(1.0),
                'variance': np.log(1.0),
            }, name='RBF_{i}'.format(i=latent), train=True)

            self.kernel_arr.append(kernel)

            prior = KernelGaussianDistribution(init={
                'kernel': kernel
            }, name='KernelGaussianDistribtuion_{i}'.format(i=latent), train=True)

            self.prior_arr.append(prior)

            approx_posterior = WhitenedGaussianDistribution(init={
                'mean': np.zeros(N)[:, None], 
                'covariance_chol': [init['covariance_chol'], init['N']],
                'kernel': kernel,
                'Z': self.Z[latent],
            }, name='KernelGaussianDistribtuion_{i}'.format(i=latent), train=True)
            self.approx_posterior_arr.append(approx_posterior)

