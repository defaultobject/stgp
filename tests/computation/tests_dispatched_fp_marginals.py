""" Unittests for dispatched full-posterior marginals. """
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

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('lik_var', [0.2])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
def test__full_approximate_posterior_marginal(seed, N, full_posterior_joint_model_no_sparsity):
    # ==== Arrange ====
    q, likelihood, prior, sparsity, data = full_posterior_joint_model_no_sparsity

    # ==== Act ====
    v_fn = evoke('variational_params', q, likelihood, prior)

    q_m, q_S_chol = v_fn(
        data, q, likelihood, prior 
    )

    marginal_fn = evoke('marginal', q, likelihood, prior)

    marginal_m, marginal_S = marginal_fn(
        data, q_m, q_S_chol, q, likelihood, prior
    )

    # === Assert ===
    # so sparsity is used so the variational params should be unchanged
    m_true = q.m
    S_true = q.S_diag[..., None]

    np.testing.assert_allclose(m_true, marginal_m)
    np.testing.assert_allclose(S_true, marginal_S)
