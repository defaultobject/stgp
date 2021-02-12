import unittest

import sys
sys.path.append('../../')

import gpjax
from gpjax.computation import gaussian_expected_log_likelihood

import numpy as np
import numpy.testing as npt
import scipy

class TestExpectedLogLikelihood(unittest.TestCase):
    def test_is_scalar(self):
        N = 10

        Y = np.ones(N)[:, None]
        noise = 0.1
        q_mu = np.ones(N)[:, None]
        q_covar_diag = np.ones(N)[:, None]

        ell = gaussian_expected_log_likelihood(Y, noise, q_mu, q_covar_diag)

        self.assertTrue(
            len(ell.shape)==0
        )

    def test_deterministic_q(self):
        N = 1

        Y = np.ones(N)[:, None]
        noise = 0.001
        q_mu = np.ones(N)[:, None]
        q_covar_diag = np.zeros(N)[:, None]

        ell = gaussian_expected_log_likelihood(Y, noise, q_mu, q_covar_diag)

        true_ell = scipy.stats.multivariate_normal.logpdf(Y, q_mu, np.eye(Y.shape[0])*noise)

        npt.assert_almost_equal(ell, true_ell, decimal=5)
        
    def test_deterministic_q_n_50(self):
        N = 50

        np.random.seed(0)
        Y = np.random.randn(N)[:, None]
        noise = 0.1
        q_mu = np.random.randn(N)[:, None]
        q_covar_diag = np.zeros(N)[:, None]

        ell = gaussian_expected_log_likelihood(Y, noise, q_mu, q_covar_diag)

        true_ell = scipy.stats.multivariate_normal.logpdf(Y[:, 0], q_mu[:, 0], np.eye(Y.shape[0])*noise)

        npt.assert_almost_equal(ell, true_ell, decimal=5)

    def test_n_50(self):
        N = 10

        np.random.seed(0)
        Y = np.random.randn(N)[:, None]
        noise = 0.1
        q_mu = np.random.randn(N)[:, None]
        q_covar_diag = 100*np.random.randn(N)[:, None]

        ell = gaussian_expected_log_likelihood(Y, noise, q_mu, q_covar_diag)

        true_ell = scipy.stats.multivariate_normal.logpdf(Y[:, 0], q_mu[:, 0], np.eye(Y.shape[0])*noise) -0.5*(1/noise)*np.sum(q_covar_diag)

        npt.assert_almost_equal(ell, true_ell, decimal=3)

if __name__ == '__main__':
    unittest.main()

