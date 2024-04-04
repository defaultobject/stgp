""" Unittests for recovering batch models. """
import jax 
import jax.numpy as jnp
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', True)
import objax

import pytest
import numpy as np
import scipy

import stgp
from stgp import settings
from stgp.models import GP
from stgp.computation.elbos.kullback_leiblers import gaussian_cholesky_kl, gaussian_kl, whitened_gaussian_kl
from stgp.dispatch import evoke
from stgp.sparsity import NoSparsity
from stgp.approximate_posteriors import FullGaussianApproximatePosterior, MeanFieldApproximatePosterior
from stgp.kernels import RBF, ScaleKernel
from stgp.likelihood import Gaussian
from stgp.kernels.diff_op import SecondOrderDerivativeKernel_2D
from stgp.trainers import NatGradTrainer

from ..common_fixtures import regression_1d_data, rbf_1d_kernel, gaussian_likelihood, full_posterior_joint_model_no_sparsity, gp_prior_1d, gaussian_likelihood, gp_model_1d, multi_output_timeseries
from ..timeseries_model_fixtures import vgp, gp, lmc, vi_fp_lmc


@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [50])
@pytest.mark.parametrize('NS', [100])
@pytest.mark.parametrize('lik_var', [0.2])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
@pytest.mark.parametrize('whiten', [True, False])
def test__gp_and_vgp_match_after_natgrad(seed, N, NS, vgp, gp):
    """
    After a natural gradient a variational GP and batch GP are equivalent
    """
    settings.jitter = 1e-5
    settings.ng_jitter = 1e-7
    # ==== Arrange ====
    XS = np.linspace(0, 1, NS)[:, None]

    # ==== Act ====
    ng_trainer = NatGradTrainer(vgp)
    ng_trainer.train(1.0, 1) 

    vgp_elbo = vgp.get_objective()
    gp_elbo = gp.get_objective()

    # collect predictions
    vgp_pred_mu, vgp_pred_var = vgp.predict_y(XS)
    gp_pred_mu, gp_pred_var = gp.predict_y(XS)

    # === Assert  ===
    np.testing.assert_allclose(vgp_elbo, gp_elbo, rtol=1e-5)
    np.testing.assert_allclose(np.squeeze(gp_pred_mu), np.squeeze(vgp_pred_mu), rtol=1e-3)
    np.testing.assert_allclose(np.squeeze(gp_pred_var), np.squeeze(vgp_pred_var), rtol=1e-3)


@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('P', [2])
@pytest.mark.parametrize('N', [50])
@pytest.mark.parametrize('NS', [100])
@pytest.mark.parametrize('lik_var', [0.2])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
@pytest.mark.parametrize('whiten', [True, False])
def test__lmc_and_vi_fp_lmc_match_after_natgrad(seed, P, N, NS, vi_fp_lmc, lmc):
    """
    After a natural gradient a variational LMC and batch LMC are equivalent
    """
    settings.jitter = 1e-5
    settings.ng_jitter = 1e-7
    # ==== Arrange ====
    XS = np.linspace(0, 1, NS)[:, None]

    # ==== Act ====
    ng_trainer = NatGradTrainer(vi_fp_lmc)
    obj_val, _ = ng_trainer.train(1.0, 1) 

    vgp_elbo = vi_fp_lmc.get_objective()
    gp_elbo = lmc.get_objective()

    # collect predictions
    vgp_pred_mu, vgp_pred_var = vi_fp_lmc.predict_y(XS)
    gp_pred_mu, gp_pred_var = lmc.predict_y(XS)

    breakpoint()

    # === Assert  ===
    np.testing.assert_allclose(vgp_elbo, gp_elbo, rtol=1e-4)
    np.testing.assert_allclose(np.squeeze(gp_pred_mu), np.squeeze(vgp_pred_mu), rtol=1e-2)
    np.testing.assert_allclose(np.squeeze(gp_pred_var), np.squeeze(vgp_pred_var), rtol=1e-2)


