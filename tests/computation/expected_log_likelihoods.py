"""
Unittests for implemented log Gaussians. To rest we approximate E[log P(Y|f)] with quadrature + monte-carlo
    and assert that it is close to the implemented version.
"""

import pytest

import gpax
from gpax.sparsity import NoSparsity
from gpax.computation.expected_log_likelihoods import gaussian_expected_log_likelihood

import numpy as np

from ..common_fixtures import regression_1d_data, gaussian_likelihood, rbf_1d_kernel, gaussian_approximate_posterior



@pytest.mark.parametrize('N', [30])
@pytest.mark.parametrize('var', [0.3])
@pytest.mark.parametrize('lengthscale', [0.7])
def test_expected_log_likelihood(N, regression_1d_data, gaussian_likelihood, rbf_1d_kernel, gaussian_approximate_posterior):
    X, Y = regression_1d_data
    lik = gaussian_likelihood
    kernel = rbf_1d_kernel
    q = gaussian_approximate_posterior

    #q_f_m, q_f_s = q.marginal(X, kernel, NoSparsity(X))
    q_f_m, q_f_s = q.m, q.S_diag
    ell_test = gaussian_expected_log_likelihood(X, Y, lik.variance, q_f_m, q_f_s)

    N_samples = 50000
    samples = np.random.normal(np.squeeze(q_f_m), np.sqrt(np.squeeze(q_f_s)), size= (N_samples, N))

    val = 0.0
    for s in range(N_samples):
        val += np.sum(lik.log_likelihood(Y, samples[s][:, None]))

    ell_true = val/N_samples

    assert ell_true == ell_test
    np.allclose(ell_true, ell_test)

