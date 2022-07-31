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

from ..common_fixtures import regression_1d_data, rbf_1d_kernel, gaussian_likelihood, full_posterior_joint_model_no_sparsity, gp_prior_1d, gaussian_likelihood, gp_model_1d

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('NS', [100])
@pytest.mark.parametrize('lik_var', [0.2])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
def test__gp_prediction(seed, N, NS, gp_model_1d):
    # ==== Arrange ====
    XS = np.linspace(0, 1, NS)[:, None]
    m = gp_model_1d

    # ==== Act ====
    f_mu_diag, f_var_diag = m.predict_f(XS, diagonal=True, squeeze=False)
    f_mu_full, f_var_full = m.predict_f(XS, diagonal=False, squeeze=False)

    y_mu_diag, y_var_diag = m.predict_y(XS, diagonal=True, squeeze=False)
    y_mu_full, y_var_full = m.predict_y(XS, diagonal=False, squeeze=False)

    # === Assert -- diagonals ===
    np.testing.assert_equal(
        f_mu_diag.shape,
        [1, NS]
    )

    np.testing.assert_equal(
        f_var_diag.shape,
        [1, NS]
    )

    np.testing.assert_equal(
        y_mu_diag.shape,
        [1, NS]
    )

    np.testing.assert_equal(
        y_var_diag.shape,
        [1, NS]
    )

    # === Assert -- full ===
    np.testing.assert_equal(
        f_mu_full.shape,
        [1, NS]
    )

    np.testing.assert_equal(
        f_var_full.shape,
        [1, NS, NS]
    )

    np.testing.assert_equal(
        y_mu_full.shape,
        [1, NS]
    )

    np.testing.assert_equal(
        y_var_full.shape,
        [1, NS, NS]
    )
