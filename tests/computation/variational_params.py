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

from ..common_fixtures import regression_2d_data, rbf_2d_kernel, gaussian_likelihood, full_posterior_joint_model_no_sparsity, gp_prior_2d

import tensorflow
import tensorflow_probability as tfp

@pytest.fixture
def gaussian_dist_mean_var(seed, N):
    np.random.seed(seed)

    m = np.random.randn(N, 1)
    S_chol_vec = np.random.randn(int(N*(N-1)/2))

    S_chol = np.tril(S_chol_vec, N)

    return m, S_chol

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('lik_var', [0.2])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
def test__full_approximate_posterior(seed, N, full_posterior_joint_model_no_sparsity):
    # ==== Arrange ====

    q, likelihood, prior, sparsity, data = full_posterior_joint_model_no_sparsity


    # ==== Act ====
    v_fn = evoke('variational_params', q, likelihood, prior)

    m_test, S_chol_test = v_fn(
        data, q, likelihood, prior 
    )

    # === Assert ===
    # so sparsity is used so the variational params should be unchanged
    m_true = q.m
    S_true = q.S_chol

    np.testing.assert_allclose(m_true, m_test)
    np.testing.assert_allclose(S_true, S_chol_test)
