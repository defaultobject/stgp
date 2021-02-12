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

from gpjax.computation.conditionals import *

import tensorflow as tf

import numpy as np
import numpy.testing as npt
import scipy
from scipy.stats import random_correlation
import gpflow 
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

class TestConditionals(unittest.TestCase):
    def test_kernel_gaussian_gaussian_conditional_diagional__same_as_gpflow(self):
        np.random.seed(0)
        #setup
        X = np.linspace(0, 1, 100)[:, None]
        Xs = np.linspace(-5, 0, 200)[:, None]
        
        N = 100

        m_rand = np.random.randn(N)[:, None]
        _, S_sqrt, S = get_random_correlation_matrix(N)

        #m_rand=np.zeros(N)[:, None]
        #S = np.eye(N)
        #S_sqrt = np.linalg.cholesky(S)

        #create gpflow conditional
        K_ls = 5.245
        K_var = 3.0

        K = gpflow.kernels.RBF(lengthscales=K_ls, variance=K_var)
        mu = m_rand
        q_sqrt = S_sqrt[np.tril_indices(N, 0)]

        full_output_cov = False
        full_cov = False
        white = False

        m = gpflow.models.SVGP(K, gpflow.likelihoods.Gaussian(), X, num_data=N, whiten=False, q_mu=mu, q_sqrt=np.array([S_sqrt]))

        #create gpjax conditional
        gpjax.settings.Settings.jitter = gpflow.default_jitter()
        K_gpjax = gpjax.kernels.RBF(lengthscales=np.array([np.log(K_ls)]), variances=np.array([np.log(K_var)]))
        g_posterior = gpjax.distributions.GaussianDistribution(mu=mu, covariance_chol=q_sqrt, meta={'N': N})
        g_prior = gpjax.distributions.KernelGaussianDistribution(kernel=K_gpjax, meta={'N': N, 'X': X})

        #run
        true_mu, true_var = m.predict_f(Xs, full_cov=False)
        test_mu, test_var = kernel_gaussian_gaussian_conditional_diagional(Xs, X, g_prior, g_posterior)

        #assert
        npt.assert_allclose(true_mu, test_mu)
        npt.assert_allclose(true_var, test_var)
        
    def test_kernel_gaussian_gaussian_conditional__same_as_gpflow(self):
        np.random.seed(0)
        #setup
        X = np.linspace(0, 1, 100)[:, None]
        Xs = np.linspace(-5, 0, 200)[:, None]
        
        N = 100

        K_ls = 5.245
        K_var = 3.0

        m_rand = np.random.randn(N)[:, None]
        _, S_sqrt, S = get_random_correlation_matrix(N)

        #m_rand=np.zeros(N)[:, None]
        #S = np.eye(N)
        #S_sqrt = np.linalg.cholesky(S)

        #create gpflow conditional
        K = gpflow.kernels.RBF(lengthscales=K_ls, variance=K_var)
        mu = m_rand
        q_sqrt = S_sqrt[np.tril_indices(N, 0)]

        full_output_cov = False
        full_cov = False
        white = False

        m = gpflow.models.SVGP(K, gpflow.likelihoods.Gaussian(), X, num_data=N, whiten=False, q_mu=mu, q_sqrt=np.array([S_sqrt]))

        #create gpjax conditional
        gpjax.settings.Settings.jitter = gpflow.default_jitter()
        K_gpjax = gpjax.kernels.RBF(lengthscales=np.array([np.log(K_ls)]), variances=np.array([np.log(K_var)]))
        g_posterior = gpjax.distributions.GaussianDistribution(mu=mu, covariance_chol=q_sqrt, meta={'N': N})
        g_prior = gpjax.distributions.KernelGaussianDistribution(kernel=K_gpjax, meta={'N': N, 'X': X})

        #run
        true_mu, true_var = m.predict_f(Xs, full_cov=True)
        true_var = true_var[0, ...]

        test_mu, test_var = kernel_gaussian_gaussian_conditional(Xs, X, g_prior, g_posterior)

        #assert
        npt.assert_allclose(true_mu, test_mu)
        npt.assert_allclose(true_var, test_var)

    def test_whitened_kernel_gaussian_gaussian_conditional_diagional__same_as_gpflow(self):
        np.random.seed(0)
        #setup
        X = np.linspace(0, 1, 100)[:, None]
        Xs = np.linspace(-5, 0, 200)[:, None]
        
        N = 100

        m_rand = np.random.randn(N)[:, None]
        _, S_sqrt, S = get_random_correlation_matrix(N)

        K_ls = 5.245
        K_var = 3.0

        #m_rand=np.zeros(N)[:, None]
        #S = np.eye(N)
        #S_sqrt = np.linalg.cholesky(S)

        #create gpflow conditional
        K = gpflow.kernels.RBF(lengthscales=K_ls, variance=K_var)
        mu = m_rand
        q_sqrt = S_sqrt[np.tril_indices(N, 0)]

        full_output_cov = False
        full_cov = False
        white = True

        m = gpflow.models.SVGP(K, gpflow.likelihoods.Gaussian(), X, num_data=N, whiten=True, q_mu=mu, q_sqrt=np.array([S_sqrt]))

        #create gpjax conditional
        gpjax.settings.Settings.jitter = gpflow.default_jitter()
        K_gpjax = gpjax.kernels.RBF(lengthscales=np.array([np.log(K_ls)]), variances=np.array([np.log(K_var)]))
        g_posterior = gpjax.distributions.GaussianDistribution(mu=mu, covariance_chol=q_sqrt, meta={'N': N})
        g_prior = gpjax.distributions.KernelGaussianDistribution(kernel=K_gpjax, meta={'N': N, 'X': X})

        #run

        true_mu, true_var = m.predict_f(Xs, full_cov=False)
        test_mu, test_var = whitened_kernel_gaussian_gaussian_conditional_diagional(Xs, X, g_prior, g_posterior)

        #assert
        npt.assert_allclose(true_mu, test_mu)
        npt.assert_allclose(true_var, test_var)

    def test_whitened_kernel_gaussian_gaussian_conditional__same_as_gpflow(self):
        np.random.seed(0)
        #setup
        X = np.linspace(0, 1, 100)[:, None]
        Xs = np.linspace(-5, 0, 200)[:, None]
        
        N = 100

        K_ls = 5.245
        K_var = 3.0

        m_rand = np.random.randn(N)[:, None]
        _, S_sqrt, S = get_random_correlation_matrix(N)

        #m_rand=np.zeros(N)[:, None]
        #S = np.eye(N)
        #S_sqrt = np.linalg.cholesky(S)

        #create gpflow conditional
        K = gpflow.kernels.RBF(lengthscales=K_ls, variance=K_var)
        mu = m_rand
        q_sqrt = S_sqrt[np.tril_indices(N, 0)]

        full_output_cov = False
        full_cov = False
        white = True

        m = gpflow.models.SVGP(K, gpflow.likelihoods.Gaussian(), X, num_data=N, whiten=True, q_mu=mu, q_sqrt=np.array([S_sqrt]))

        #create gpjax conditional
        gpjax.settings.Settings.jitter = gpflow.default_jitter()
        K_gpjax = gpjax.kernels.RBF(lengthscales=np.array([np.log(K_ls)]), variances=np.array([np.log(K_var)]))
        g_posterior = gpjax.distributions.GaussianDistribution(mu=mu, covariance_chol=q_sqrt, meta={'N': N})
        g_prior = gpjax.distributions.KernelGaussianDistribution(kernel=K_gpjax, meta={'N': N, 'X': X})

        #run

        true_mu, true_var = m.predict_f(Xs, full_cov=True)
        true_var = true_var[0, ...]
        test_mu, test_var = whitened_kernel_gaussian_gaussian_conditional(Xs, X, g_prior, g_posterior)

        #assert
        npt.assert_allclose(true_mu, test_mu)
        npt.assert_allclose(true_var, test_var)
