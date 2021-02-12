from . import Model, SDE_GP

from ..distributions import GaussianDistribution, ZeroMeanGaussianDistribution, KernelGaussianDistribution, DiagionalGaussianDistribution, SparseGaussianDistribution, WhitenedGaussianDistribution
from ..likelihoods import GaussianLikelihood
from ..kernels import RBF, Matern32
from ..decorators import return_gradients
from .. import Parameter
from ..settings import Settings
from ..computation import gaussian_expected_log_likelihood
from ..computation.kalman_filter import kalman_loop, kalman_loop_store_intermediate


import jax
import jax.numpy as np
import numpy as onp
from jax import value_and_grad

from jax.config import config
config.update("jax_enable_x64", True)

import json

import matplotlib.pyplot as plt

class SDE_VGP(SDE_GP):
    def __init__(self, X, Y, params, meta):
        self.X = X
        self.Y = Y
        self.Z = X

        N = meta['N']
        self.N = N
        M = self.Z.shape[0]

        if params is None:
            params = {
                'N': N,
                'covariance_chol': 0.01*np.ones(int(M*(M+1)/2))
            }

        self.kernel = Matern32(params={
            'lengthscale': np.log(1.0),
            'variance': np.log(1.0)
        })

        self.prior = KernelGaussianDistribution(meta={
            'kernel': self.kernel
        })


        self.approx_posterior = GaussianDistribution(
            params = params['approx_posterior'],
            meta = {
                'kernel': self.kernel,
                'N': N
            }
        )

        self.likelihood = GaussianLikelihood(params={'variance':np.log(0.01)})
        self.setup()

    def get_KL(self):
        #alpha = [ K_xx + jit*I]^{-1} m
        #beta = [K_xx +jit*I]^{-1/2} m
        #chol_diag = diag([K + jit*I]^{1/2})

        S_sqrt = self.approx_posterior.covar_sqrt(None, None)
        m = self.approx_posterior.mean(None)
        M = m .shape[0]

        alpha, beta, chol_diag, A = self.get_computational_primitives(self.X, m, S_sqrt)

        beta = beta[:, None] #M x 1

        log_dets = np.sum(2*np.log(chol_diag)) - np.sum(2*np.log(np.diag(S_sqrt)))

        solve_m = np.sum(A)

        solve_v = m.T @ alpha

        KL =  log_dets - M + solve_m + solve_v[0]

        return KL

    @return_gradients
    def get_objective(self):        
        if True:
            ell = self.likelihood.expected_log_likelihood(self.X, self.Y, self.approx_posterior, return_grad=False)
        else:
            m = self.approx_posterior.mean(None)
            S_sqrt = self.approx_posterior.covar_sqrt(None, None)

            S_diag = np.sum(np.multiply(S_sqrt, S_sqrt), axis=1)

            ell = gaussian_expected_log_likelihood(self.Y, self.likelihood.variance, m, S_diag)

        if False:
            kl = self.approx_posterior.KL(self.prior, X=self.Z, return_grad=False)
        else:
            kl = self.get_KL()


        elbo =  ell-kl 
        return -elbo

    def predict(self, X):
        mu, sig = self.approx_posterior.predict(X)

        return mu, sig
