import unittest

import sys
sys.path.append('../../')

import jax 
import jax.numpy as jnp

import gpjax
from gpjax.computation.gaussian import log_gaussian
from gpjax.kernels import Matern32
from gpjax.likelihoods import GaussianLikelihood

import numpy as np
import numpy.testing as npt
import scipy
from scipy.stats import random_correlation

import os
import subprocess


class TestGaussian(unittest.TestCase):
    def test_log_gaussian__zero_mean__kernel_sigma__same_as_scipy(self):
        #result
        np.random.seed(0)

        N = 10
        x = np.linspace(0, 1, N)[:, None]
        K_xx = Matern32().K(x, x)

        Y = np.random.randn(N)[:, None]
        mu = np.zeros_like(Y)

        #result
        res = log_gaussian(Y, mu, K_xx)

        #check  
        true = scipy.stats.multivariate_normal(mu[:, 0], K_xx).logpdf(Y[:, 0])

        npt.assert_almost_equal(res, true, decimal=5)

    def test_log_gaussian__kernel_sigma__same_as_scipy(self):
        #result
        np.random.seed(0)

        N = 10
        x = np.linspace(0, 1, N)[:, None]
        K_xx = Matern32().K(x, x)

        Y = np.random.randn(N)[:, None]
        mu = np.random.randn(N)[:, None]

        #result
        res = log_gaussian(Y, mu, K_xx)

        #check  
        true = scipy.stats.multivariate_normal(mu[:, 0], K_xx).logpdf(Y[:, 0])

        npt.assert_almost_equal(res, true, decimal=5)

    def test_log_gaussian__kernel_sigma__large__same_as_scipy(self):
        #result
        np.random.seed(0)

        N = 100
        x = np.linspace(0, 10, N)[:, None]
        K_xx = Matern32().K(x, x)

        Y = np.random.randn(N)[:, None]
        mu = np.random.randn(N)[:, None]

        #result
        res = log_gaussian(Y, mu, K_xx)

        #check  
        true = scipy.stats.multivariate_normal(mu[:, 0], K_xx).logpdf(Y[:, 0])

        npt.assert_almost_equal(res, true, decimal=5)

if __name__ == '__main__':
    with jax.disable_jit():
        unittest.main()


