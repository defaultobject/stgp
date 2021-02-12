from . import Model

from ..distributions import GaussianDistribution, ZeroMeanGaussianDistribution, KernelGaussianDistribution, DiagionalGaussianDistribution, SparseGaussianDistribution, WhitenedGaussianDistribution
from ..likelihoods import GaussianLikelihood
from ..kernels import RBF, Matern32
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


class VGP(Model):
    def __init__(self, X, Y, params, meta, num_latents=None):
        """
            @TODO: default behaviours:
                if Y in N x P then have P independent latent functions
                if have num_latents and P = 1 then just have P versions of the same GP
                if have num_latents != P && P > 1 then throw run time error
        """
        if params==None:
            params = {}

        self.X = X
        self.Y = Y
        self.Z = X
        self.num_latents = num_latents
        self.params = params

        if self.num_latents is None:
            self.num_latents = 1
            self.X = [self.X]
            self.Y = [self.Y]
            self.Z = [self.Z]
        
        self.kernel_arr = []
        self.prior_arr = []
        self.approx_posterior_arr = []

        N = meta['N']

        likelihood_params = self.params['likelihood']
        if type(likelihood_params) is dict:
            likelihood_params = [likelihood_params for latent in range(self.num_latents)]

        self.likelihood = [GaussianLikelihood(params=likelihood_params[latent]) for latent in range(self.num_latents)]


        kernel_params = self.params['kernel']
        if type(kernel_params) is dict:
            kernel_params = [kernel_params for latent in range(self.num_latents)]

        for latent in range(self.num_latents):
            #kernel = RBF(params=kernel_params[latent], name='RBF_{i}'.format(i=latent))
            kernel = Matern32(params={
                'lengthscale': np.log(1.0),
                'variance': np.log(1.0)
            }, name='Matern32_{i}'.format(i=latent))

            prior = KernelGaussianDistribution(meta={
                'kernel': kernel
            })

            self.prior_arr.append(prior)

            approx_posterior = WhitenedGaussianDistribution(
                params={
                    'mean': params['approx_posterior']['mean'], 
                    'covariance_chol': params['approx_posterior']['covariance_chol'],
                    'Z': self.Z[latent]
                }, 
                meta = {
                    'kernel': kernel,
                    'N': meta['N']
                }
            )

            self.approx_posterior_arr.append(approx_posterior)

    @return_gradients
    def get_objective(self):
        elbo = 0.0
        for latent in range(self.num_latents):
            ell = self.likelihood[latent].expected_log_likelihood(self.X[latent], self.Y[latent], self.approx_posterior_arr[latent], return_grad=False)
            kl = self.approx_posterior_arr[latent].KL(self.prior_arr[latent], X=self.Z[latent], return_grad=False)

            elbo_l =  ell-kl 
            elbo += elbo_l
        return -elbo

    def predict(self, X):
        mu_arr = []
        sig_arr = []

        for latent in range(self.num_latents):
            mu, sig = self.approx_posterior_arr[latent].predict(X)

            mu_arr.append(mu)
            sig_arr.append(sig)

        if self.num_latents is 1:
            return mu_arr[0], sig_arr[0]

        return mu_arr, sig_arr




