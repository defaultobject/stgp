""" Unittests for negative log likelihoods """
import jax 
import jax.numpy as jnp
from jax.config import config as jax_config
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
from stgp.computation.gaussian import log_gaussian, log_gaussian_diagonal

from ..common_fixtures import regression_1d_data, rbf_1d_kernel, gaussian_likelihood, full_posterior_joint_model_no_sparsity, gp_prior_1d, gaussian_likelihood, gp_model_1d, multi_output_timeseries
from ..timeseries_model_fixtures import vgp, gp, lmc, vi_fp_lmc


@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [50])
@pytest.mark.parametrize('NS', [100])
@pytest.mark.parametrize('lik_var', [0.2])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
@pytest.mark.parametrize('whiten', [False])
def test__vgp_nlpd(seed, N, NS, vgp):
    """ Test variational gp nlpd.  """
    X, Y = vgp.data.X, vgp.data.Y
    # ==== Act ====

    # train so the nlpd is not awful
    ng_trainer = NatGradTrainer(vgp)
    ng_trainer.train(1.0, 1) 

    # collect predictions
    vgp_pred_mu, vgp_pred_var = vgp.predict_y(X)

    # Compute true nlpd
    true_nlpd = -np.mean(log_gaussian_diagonal(Y, vgp_pred_mu, vgp_pred_var))

    # Get vgp nlpd
    vgp_nlpd = vgp.nlpd(X, Y, num_samples=10000)
    vgp_nlpd = np.squeeze(vgp_nlpd[0])

    # === Assert  ===
    np.testing.assert_allclose(true_nlpd, vgp_nlpd, rtol=1e-4)

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('P', [2])
@pytest.mark.parametrize('N', [50])
@pytest.mark.parametrize('NS', [100])
@pytest.mark.parametrize('lik_var', [0.2])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
@pytest.mark.parametrize('whiten', [True, False])
def test__vi_fp_lmc_nlpd(seed, P, N, NS, vi_fp_lmc):
    """ Test variational fp LMC nlpd.  """
    # ==== Act arrang====
    X, Y = vi_fp_lmc.data.X, vi_fp_lmc.data.Y

    # ==== Act ====

    # train so the nlpd is not awful
    ng_trainer = NatGradTrainer(vi_fp_lmc)
    ng_trainer.train(1.0, 1) 

    # collect predictions
    pred_mu, pred_var = vi_fp_lmc.predict_y(X)

    # Compute true nlpd
    true_nlpd = - np.mean(jax.vmap(
        log_gaussian_diagonal, 
        [1, 1, 1]
    )(
        Y[..., None], 
        pred_mu[..., None], 
        pred_var[..., None]
    ), axis=1)

    # Get vi_fp_lmc nlpd
    vi_fp_lmc_nlpd = vi_fp_lmc.nlpd(X, Y, num_samples=20000)
    vi_fp_lmc_nlpd = np.squeeze(vi_fp_lmc_nlpd[0])

    # === Assert  ===
    np.testing.assert_allclose(true_nlpd, vi_fp_lmc_nlpd, rtol=1e-3)

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('P', [2])
@pytest.mark.parametrize('N', [50])
@pytest.mark.parametrize('NS', [100])
@pytest.mark.parametrize('lik_var', [0.2])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
@pytest.mark.parametrize('whiten', [True, False])
def test__vi_fp_lmc_global_nlpd(seed, P, N, NS, vi_fp_lmc):
    """ Test variational fp LMC nlpd.  """
    # ==== Act arrang====
    X, Y = vi_fp_lmc.data.X, vi_fp_lmc.data.Y

    # ==== Act ====

    # train so the nlpd is not awful
    ng_trainer = NatGradTrainer(vi_fp_lmc)
    ng_trainer.train(1.0, 1) 

    # collect predictions
    pred_mu, pred_var = vi_fp_lmc.predict_y(X, diagonal=False)

    # Compute true nlpd
    true_nlpd = - np.mean(jax.vmap(
        log_gaussian, 
        [0, 0, 0]
    )(
        Y[..., None], 
        pred_mu[..., None], 
        pred_var
    ))

    # Get vi_fp_lmc nlpd
    vi_fp_lmc_nlpd = vi_fp_lmc.nlpd(X, Y, num_samples=20000)
    vi_fp_lmc_nlpd = np.squeeze(vi_fp_lmc_nlpd[1])

    # === Assert  ===
    np.testing.assert_allclose(true_nlpd, vi_fp_lmc_nlpd, rtol=1e-3)




