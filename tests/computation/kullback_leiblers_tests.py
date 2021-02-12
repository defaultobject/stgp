import unittest

import sys
sys.path.append('../../')

import jax 
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_disable_jit', True)
import jax.numpy as jnp

import gpjax
from gpjax.computation import positive_transform, lower_triangle, cholesky_solve,log_chol_matrix_det, correlation_transform, get_correlation_cholesky, kf_cholesky_solve_trace
from gpjax.kernels import Matern32
from gpjax.likelihoods import GaussianLikelihood
from gpjax.distributions import *
from gpjax.approximate_posteriors import *
from gpjax.inference import *

from gpjax.computation.kullback_leiblers import *
import gpflow
import tensorflow as tf

import numpy as np
import numpy.testing as npt
import scipy
from scipy.stats import random_correlation
import os
import subprocess

def get_random_correlation_matrix(N, jit=1e-8):
    num_random_elements = int(N*(N+1)/2)
    L = np.zeros([N, N])
    a = np.random.randn(num_random_elements)
    L[np.tril_indices(N, 0)] = a
    A = L @ L.T

    #add jitter
    A = A + jit*np.eye(N)
    L = np.linalg.cholesky(A)
    a = L[np.tril_indices(N, 0)]
    return a,  L, L @ L.T

class TestKullbackLeiblers(unittest.TestCase):
    def test_gaussian_kl__same_as__gpflow(self):
        np.random.seed(0)
        #setup
        X = np.linspace(0, 1, 100)[:, None]
        Xs = np.linspace(-5, 0, 200)[:, None]
        
        N = 100
        Y = np.random.randn(N)[:, None]

        m_rand = np.random.randn(N)[:, None]
        _, S_sqrt, S = get_random_correlation_matrix(N)

        K_ls = 0.4
        K_var = 3.4

        #create gpflow objects
        K = gpflow.kernels.RBF(lengthscales=K_ls, variance=K_var)
        mu = m_rand
        q_sqrt = S_sqrt[np.tril_indices(N, 0)]

        full_output_cov = False
        full_cov = False
        white = False

        m = gpflow.models.SVGP(K, gpflow.likelihoods.Gaussian(), X, num_data=N, whiten=False, q_mu=mu, q_sqrt=np.array([S_sqrt]))


        #create gpjax objects
        gpjax.settings.Settings.jitter = gpflow.default_jitter()
        K_gpjax = gpjax.kernels.RBF(lengthscales=np.array([np.log(K_ls)]), variances=np.array([np.log(K_var)]))
        g_likelihood = gpjax.likelihoods.GaussianLikelihood()
        g_posterior = gpjax.distributions.GaussianDistribution(mu=mu, covariance_chol=q_sqrt, meta={'N': N})
        g_prior = gpjax.distributions.KernelGaussianDistribution(kernel=K_gpjax, meta={'N': N, 'X': X})
        g_q = GaussianApproxPosterior(dim=N, m=m_rand, S_chol=q_sqrt)
        g_q = MeanFieldApproxPosterior(components=[g_q])

        inference = VariationalInference()
        options = {
            'whiten': False
        }
        inference.initialize(g_q)
        m_jax = gpjax.models.GP(X, Y, K_gpjax, inference=inference, options=options)

        #compare
        kl_true = np.array(m.prior_kl())
        kl_test = np.array(m_jax.kl())

        #assert
        npt.assert_allclose(kl_true, kl_test)

    def test_whitened_gaussian_kl__same_as__gpflow(self):
        np.random.seed(0)
        #setup
        X = np.linspace(0, 1, 100)[:, None]
        Xs = np.linspace(-5, 0, 200)[:, None]
        
        N = 100
        Y = np.random.randn(N)[:, None]

        m_rand = np.random.randn(N)[:, None]
        _, S_sqrt, S = get_random_correlation_matrix(N)

        K_ls = 0.4
        K_var = 3.4

        #create gpflow objects
        K = gpflow.kernels.RBF(lengthscales=K_ls, variance=K_var)
        mu = m_rand
        q_sqrt = S_sqrt[np.tril_indices(N, 0)]

        full_output_cov = False
        full_cov = False
        white = True

        m = gpflow.models.SVGP(K, gpflow.likelihoods.Gaussian(), X, num_data=N, whiten=True, q_mu=mu, q_sqrt=np.array([S_sqrt]))


        #create gpjax objects
        gpjax.settings.Settings.jitter = gpflow.default_jitter()
        K_gpjax = gpjax.kernels.RBF(lengthscales=np.array([np.log(K_ls)]), variances=np.array([np.log(K_var)]))
        g_likelihood = gpjax.likelihoods.GaussianLikelihood()
        g_posterior = gpjax.distributions.GaussianDistribution(mu=mu, covariance_chol=q_sqrt, meta={'N': N})
        g_prior = gpjax.distributions.KernelGaussianDistribution(kernel=K_gpjax, meta={'N': N, 'X': X})
        g_q = GaussianApproxPosterior(dim=N, m=m_rand, S_chol=q_sqrt)
        g_q = MeanFieldApproxPosterior(components=[g_q])

        inference = VariationalInference()
        options = {
            'whiten': True
        }
        inference.initialize(g_q)
        m_jax = gpjax.models.GP(X, Y, K_gpjax, inference=inference, options=options)

        #compare
        kl_true = np.array(m.prior_kl())
        kl_test = np.array(m_jax.kl())

        #assert
        npt.assert_allclose(kl_true, kl_test)

if __name__ == '__main__':
    with jax.disable_jit():
        unittest.main()



