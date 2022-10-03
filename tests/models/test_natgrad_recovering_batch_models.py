""" Unittests for recovering batch models. """
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

from ..common_fixtures import regression_1d_data, rbf_1d_kernel, gaussian_likelihood, full_posterior_joint_model_no_sparsity, gp_prior_1d, gaussian_likelihood, gp_model_1d, multi_output_timeseries

@pytest.fixture
def vgp(regression_1d_data, lik_var, rbf_ls, rbf_var, whiten):
    X, Y = regression_1d_data
    data = stgp.data.Data(X, Y)


    # Create Model
    m = stgp.models.GP(
        data = data, 
        kernel=ScaleKernel(RBF(lengthscales=[rbf_ls]), rbf_var),
        likelihood = [Gaussian(variance=lik_var)],
        inference='Variational',
        whiten=whiten
    )

    return m

@pytest.fixture
def gp(regression_1d_data, lik_var, rbf_ls, rbf_var):
    X, Y = regression_1d_data
    data = stgp.data.Data(X, Y)

    # Create Model
    m = stgp.models.GP(
        data = data, 
        kernel=ScaleKernel(RBF(lengthscales=[rbf_ls]), rbf_var),
        likelihood = [Gaussian(variance=lik_var)]
    )

    return m

@pytest.fixture
def lmc(P, multi_output_timeseries, lik_var, rbf_ls, rbf_var):
    # assuming full rank
    Q = P

    X, Y = multi_output_timeseries
    data = stgp.data.Data(X, Y)

    
    # Construct Latent GPs
    Z = [stgp.sparsity.NoSparsity(X) for q in range(Q)]

    latent_kernels = [ScaleKernel(RBF(lengthscales=[rbf_ls]), rbf_var) for q in range(Q)]
    latent_gps = [
        stgp.models.GP(sparsity=Z[q], kernel=latent_kernels[q]) for q in range(Q)
    ] 

    np.random.seed(0)
    W = np.random.randn(P, Q)
    prior = stgp.transforms.multi_output.LMC(latent_gps, W = W, output_dim = P)

    # Create Model
    m = stgp.models.GP(
        data = data, 
        prior = prior
    )

    return m

@pytest.fixture
def vi_fp_lmc(P, multi_output_timeseries, lik_var, rbf_ls, rbf_var, whiten):
    # assuming full rank
    Q = P

    X, Y = multi_output_timeseries
    data = stgp.data.Data(X, Y)

    
    # Construct Latent GPs
    Z = [stgp.sparsity.NoSparsity(X) for q in range(Q)]

    latent_kernels = [ScaleKernel(RBF(lengthscales=[rbf_ls]), rbf_var) for q in range(Q)]
    latent_gps = [
        stgp.models.GP(sparsity=Z[q], kernel=latent_kernels[q]) for q in range(Q)
    ] 

    np.random.seed(0)
    W = np.random.randn(P, Q)
    prior = stgp.transforms.multi_output.LMC(latent_gps, W = W, output_dim = P)

    # Create Model
    m = stgp.models.GP(
        data = data, 
        prior = prior,
        inference='Variational',
        approximate_posterior = FullGaussianApproximatePosterior(dim = X.shape[0] * prior.base_prior.output_dim),
        whiten=whiten

    )

    return m

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
    np.testing.assert_allclose(gp_pred_mu, vgp_pred_mu, rtol=1e-3)
    np.testing.assert_allclose(gp_pred_var, vgp_pred_var, rtol=1e-3)


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

    # === Assert  ===
    np.testing.assert_allclose(vgp_elbo, gp_elbo, rtol=1e-4)
    np.testing.assert_allclose(gp_pred_mu, vgp_pred_mu, rtol=1e-2)
    np.testing.assert_allclose(gp_pred_var, vgp_pred_var, rtol=1e-2)

