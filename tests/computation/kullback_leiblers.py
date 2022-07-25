""" Unittests for kullback-leiblers permutatations.  """
import jax 
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', True)

import pytest
import numpy as np
import scipy

import stgp
from stgp import settings
from stgp.computation.elbos.kullback_leiblers import gaussian_cholesky_kl, gaussian_kl, whitened_gaussian_kl
from stgp.dispatch import evoke
from stgp.sparsity import NoSparsity
from stgp.transforms.pdes import DifferentialOperatorJoint
from stgp.approximate_posteriors import FullGaussianApproximatePosterior
from stgp.kernels.diff_op import SecondOrderDerivativeKernel_2D

from ..common_fixtures import regression_1d_data, regression_2d_data, gp_prior_2d, rbf_1d_kernel, rbf_2d_kernel, gaussian_approximate_posterior, gp_prior_1d

import tensorflow
import tensorflow_probability as tfp

@pytest.fixture
def kl_terms(N, gp_prior_1d, gaussian_approximate_posterior):

    Z = gp_prior_1d.sparsity.Z

    m = gaussian_approximate_posterior.m
    S = gaussian_approximate_posterior.S
    S_chol = gaussian_approximate_posterior.S_chol

    m2 = np.array(gp_prior_1d.mean(Z))
    K = np.array(gp_prior_1d.covar(Z, Z))
    K_chol = scipy.linalg.cholesky(K, lower=True)


    g1 = tfp.distributions.MultivariateNormalTriL(
        loc = m[:, 0], scale_tril = S_chol
    )
    g2 = tfp.distributions.MultivariateNormalTriL(
        loc = m2[:, 0], scale_tril = K_chol
    )

    KL_true = np.array(tfp.distributions.kl_divergence(g1, g2))

    return m, S, S_chol, m2, K, K_chol, g1, g2, KL_true

@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
def test__gaussian_cholesky_kl(N, kl_terms):
    # ==== Arrange ====
    m, S, S_chol, m2, K, K_chol, g1, g2, KL_true = kl_terms

    # ==== Act ====
    KL_test = np.array(gaussian_cholesky_kl(
        m, S_chol, m2, K_chol
    ))

    # ==== assert ====
    np.testing.assert_allclose(KL_true, KL_test)

@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
def test__gaussian_kl(N, kl_terms):
    # ==== Arrange ====
    settings.jitter = 0

    m, S, S_chol, m2, K, K_chol, g1, g2, KL_true = kl_terms
    # ==== Act ====

    KL_test = np.array(gaussian_kl(
        m, S, m2, K
    ))

    # ==== assert ====
    np.testing.assert_allclose(KL_true, KL_test)

@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
def test__dispatched_kl(N, gp_prior_1d, gaussian_approximate_posterior):
    # ==== Arrange ====
    kld_fn = evoke('kullback_leibler', gaussian_approximate_posterior, gp_prior_1d)

    Z = gp_prior_1d.sparsity.Z

    m = gaussian_approximate_posterior.m
    S = gaussian_approximate_posterior.S
    S_chol = gaussian_approximate_posterior.S_chol

    m2 = np.array(gp_prior_1d.mean(Z))
    K = np.array(gp_prior_1d.covar(Z, Z))
    K_chol = scipy.linalg.cholesky(K, lower=True)

    # ==== Act ====
    KL_true = np.array(gaussian_kl(
        m, S, m2, K
    ))

    KL_test = kld_fn(gaussian_approximate_posterior, gp_prior_1d)

    # ==== assert ====
    np.testing.assert_allclose(KL_true, KL_test)

@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
def test__dispatched_kl__full_joint(N, gp_prior_2d):
    # ==== Arrange ====
    N = N*N

    diff_op_prior = DifferentialOperatorJoint(
        gp_prior_2d,
        SecondOrderDerivativeKernel_2D(gp_prior_2d.kernel)
    )

    q = FullGaussianApproximatePosterior(dim = N * diff_op_prior.output_dim)


    kld_fn = evoke('kullback_leibler', q, diff_op_prior)

    # ==== Act ====
    KL_test = kld_fn(q, diff_op_prior)

    # ==== assert ====
    #atm just assert that the code runs


