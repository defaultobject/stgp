import unittest

import sys
sys.path.append('../../')

import gpjax
from gpjax.models import VGP

from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_disable_jit', True)
import numpy as np
import numpy.testing as npt

import gpjax
from gpjax.computation import positive_transform, lower_triangle, cholesky_solve,log_chol_matrix_det, correlation_transform, get_correlation_cholesky, kf_cholesky_solve_trace
from gpjax.kernels import Matern32
from gpjax.likelihoods import GaussianLikelihood
from gpjax.distributions import *
from gpjax.approximate_posteriors import *
from gpjax.inference import *

import tensorflow as tf
import gpflow

from scipy.stats import random_correlation

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


class TestVGP(unittest.TestCase):
    def _test_object_same_as_gpflow__whiten_false(self):
        np.random.seed(0)
        #setup
        gpjax.Settings.jitter=1e-6
        X = np.linspace(0, 1, 100)[:, None]
        Xs = np.linspace(-5, 0, 200)[:, None]
        
        N = 100
        Y = np.random.randn(N)[:, None]

        m_rand = np.random.randn(N)[:, None]
        _, S_sqrt, S = get_random_correlation_matrix(N)

        lik_noise = 0.4
        K_var = 3.0
        K_ls = 10.0
        #create gpflow objects
        K = gpflow.kernels.RBF(lengthscales=[K_ls], variance=K_var)
        mu = m_rand
        q_sqrt = S_sqrt[np.tril_indices(N, 0)]


        full_output_cov = False
        full_cov = False
        white = False

        m = gpflow.models.SVGP(K, gpflow.likelihoods.Gaussian(variance=lik_noise), X, num_data=N, whiten=False, q_mu=mu, q_sqrt=np.array([S_sqrt]))

        #create gpjax objects
        gpjax.settings.Settings.jitter = gpflow.default_jitter()
        K_gpjax = gpjax.kernels.RBF(lengthscales=np.array([np.log(K_ls)]), variances=np.array([np.log(K_var)]))

        g_likelihood = gpjax.likelihoods.GaussianLikelihood(variances=np.log(lik_noise))

        g_posterior = gpjax.distributions.GaussianDistribution(mu=mu, covariance_chol=q_sqrt, meta={'N': N})
        g_prior = gpjax.distributions.KernelGaussianDistribution(kernel=K_gpjax, meta={'N': N, 'X': X})
        g_q = GaussianApproxPosterior(dim=N, m=m_rand, S_chol=q_sqrt)
        g_q = MeanFieldApproxPosterior(components=[g_q])

        inference = VariationalInference()
        options = {
            'whiten': False
        }
        inference.initialize(g_q)
        m_jax = gpjax.models.GP(X, Y, K_gpjax, inference=inference, options=options, likelihood=g_likelihood)

        #run
        true = np.array(-m.elbo([X, Y]))
        test = np.array(m_jax.get_objective(return_grad=False))

        mu_gpflow, var_gpflow = m.predict_y(Xs)
        mu_gpflow, var_gpflow = np.array(mu_gpflow), np.array(var_gpflow)

        mu_jax, var_jax = m_jax.predict_y(Xs)
        mu_jax, var_jax = np.array(mu_jax)[0], np.array(var_jax)[0]

        print('predictions: ', mu_gpflow, mu_jax)

        if False:
            gpflow_ell = m.elbo([X, Y])+m.prior_kl()
            gpflow_kl = m.prior_kl()

            gpjax_ell = m_jax.model.ell()
            gpjax_kl = m_jax.model.kl()


        print('true: ', true, ' test: ', test)
        npt.assert_allclose(true, test)
        npt.assert_allclose(mu_gpflow, mu_jax)
        npt.assert_allclose(var_gpflow, var_jax)


    def test_object_same_as_gpflow__whiten_true(self):
        #np.random.seed(0)
        #setup
        X = np.linspace(0, 1, 100)[:, None]
        Xs = np.linspace(-5, 0, 200)[:, None]
        
        N = 100
        Y = np.random.randn(N)[:, None]

        m_rand = np.random.randn(N)[:, None]
        _, S_sqrt, S = get_random_correlation_matrix(N)

        lik_noise = 0.4
        K_var = 3.0
        K_ls = 0.3

        #create gpflow objects
        K = gpflow.kernels.RBF(lengthscales=K_ls, variance=K_var)
        mu = m_rand
        q_sqrt = S_sqrt[np.tril_indices(N, 0)]

        full_output_cov = False
        full_cov = False
        white = True

        print('mu: ', mu)

        m = gpflow.models.SVGP(K, gpflow.likelihoods.Gaussian(variance=lik_noise), X, num_data=N, whiten=True, q_mu=mu, q_sqrt=np.array([S_sqrt]))

        #create gpjax objects
        gpjax.settings.Settings.jitter = gpflow.default_jitter()
        K_gpjax = gpjax.kernels.RBF(lengthscales=np.array([np.log(K_ls)]), variances=np.array([np.log(K_var)]))
        g_likelihood = gpjax.likelihoods.GaussianLikelihood(variances=np.log(lik_noise))
        g_posterior = gpjax.distributions.GaussianDistribution(mu=mu, covariance_chol=q_sqrt, meta={'N': N})
        g_prior = gpjax.distributions.KernelGaussianDistribution(kernel=K_gpjax, meta={'N': N, 'X': X})
        g_q = GaussianApproxPosterior(dim=N, m=m_rand, S_chol=q_sqrt)
        g_q = MeanFieldApproxPosterior(components=[g_q])

        inference = VariationalInference()
        options = {
            'whiten': True
        }
        inference.initialize(g_q)
        m_jax = gpjax.models.GP(X, Y, K_gpjax, inference=inference, options=options,likelihood=g_likelihood)

        #run
        true = np.array(-m.elbo([X, Y]))
        test = np.array(m_jax.get_objective(return_grad=False))

        if False:
            gpflow_ell = m.elbo([X, Y])+m.prior_kl()
            gpflow_kl = m.prior_kl()

            gpjax_ell = m_jax.model.ell()
            gpjax_kl = m_jax.model.kl()

        mu_gpflow, var_gpflow = m.predict_y(Xs)
        mu_gpflow, var_gpflow = np.array(mu_gpflow), np.array(var_gpflow)

        mu_jax, var_jax = m_jax.predict_y(Xs)
        mu_jax, var_jax = np.array(mu_jax)[0], np.array(var_jax)[0]


        npt.assert_allclose(true, test)
        npt.assert_allclose(mu_gpflow, mu_jax)
        npt.assert_allclose(var_gpflow, var_jax)

    def __test_gradients(self):
        np.random.seed(0)

        N = 50
        x = np.linspace(0, 10, N)
        y = np.sin(x)+0.1*np.random.randn(N)

        X = x[:, None]
        Y = y[:, None]


        chol_length = int(N*(N+1)/2)

        init = np.random.randn(chol_length)

        model = VGP(X, Y, init={'covariance_chol': init, 'N': N})
        elbo, gradient = model.get_objective()
        gradients = gradient['GaussianDistribution/CovarianceChol']

        eps = 1e-5
        for i in range(chol_length):
            init_mod_1 = init.copy()
            init_mod_2 = init.copy()
            init_mod_1[i] = init_mod_1[i]+eps

            gpjax.Parameter.clear()
            model_1 = VGP(X, Y, init={'covariance_chol': init_mod_1, 'N': N})
            elbo_1 = model_1.get_objective(return_grad=False)

            gpjax.Parameter.clear()
            model_2 = VGP(X, Y, init={'covariance_chol': init_mod_2, 'N': N})
            elbo_2 = model_2.get_objective(return_grad=False)

            res = gradients[i]
            true = (elbo_1-elbo_2)/eps

            npt.assert_almost_equal(res,true)


if __name__ == '__main__':
    unittest.main()
