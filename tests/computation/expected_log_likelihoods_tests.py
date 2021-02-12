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

from gpjax.computation.expected_log_likelihoods import *

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


class TestExpectedLogLikelihoods(unittest.TestCase):
    def test_gaussian_gaussian_expected_log_likelihood__using_gaussian_approx_posterior__same_as_gpflow(self):
        np.random.seed(0)
        #setup
        X = np.linspace(0, 1, 100)[:, None]
        Xs = np.linspace(-5, 0, 200)[:, None]
        
        N = 100
        Y = np.random.randn(N)[:, None]

        m_rand = np.random.randn(N)[:, None]
        _, S_sqrt, S = get_random_correlation_matrix(N)

        #create gpflow objects
        K = gpflow.kernels.RBF(lengthscales=1.0, variance=1.0)
        mu = m_rand
        q_sqrt = S_sqrt[np.tril_indices(N, 0)]

        full_output_cov = False
        full_cov = False
        white = False

        m = gpflow.models.SVGP(K, gpflow.likelihoods.Gaussian(), X, num_data=N, whiten=False, q_mu=mu, q_sqrt=np.array([S_sqrt]))

        #create gpjax objects
        gpjax.settings.Settings.jitter = gpflow.default_jitter()
        K_gpjax = gpjax.kernels.RBF(lengthscales=np.array([np.log(1.0)]), variances=np.array([np.log(1.0)]))
        g_likelihood = gpjax.likelihoods.GaussianLikelihood()
        g_posterior = gpjax.distributions.GaussianDistribution(mu=mu, covariance_chol=q_sqrt, meta={'N': N})
        g_prior = gpjax.distributions.KernelGaussianDistribution(kernel=K_gpjax, meta={'N': N, 'X': X})
        g_q = GaussianApproxPosterior(dim=N, m=m_rand, S_chol=q_sqrt)

        #run
        true_ell = m.elbo([X, Y]) + m.prior_kl()
        test_ell = gaussian_gaussian_expected_log_likelihood(X, Y, g_prior, g_likelihood, g_q)

        true_ell = np.array(true_ell)
        test_ell = np.array(test_ell)

        #assert
        npt.assert_almost_equal(true_ell, test_ell, decimal=5)

 
