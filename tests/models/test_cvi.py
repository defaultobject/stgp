""" Unittests for CVI models """

import jax 
import jax.numpy as jnp
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', True)

import pytest
import numpy as np
import scipy

import stgp
from stgp.likelihood import Gaussian, ProductLikelihood
from stgp.data import Data, TemporalData
from stgp.transforms import Independent
from stgp.approximate_posteriors import MeanFieldConjugateGaussian, ConjugateGaussian, MeanFieldApproximatePosterior, GaussianApproximatePosterior
from stgp.kernels import RBF, ScaleKernel
from stgp.models import GP

from ..common_fixtures import regression_1d_data, rbf_1d_kernel, gaussian_likelihood, full_posterior_joint_model_no_sparsity, gp_prior_1d, gaussian_likelihood, gp_model_1d

@pytest.fixture
def cvi_gp(regression_1d_data, lik_var, rbf_ls, rbf_var):
    X, Y = regression_1d_data
    data = stgp.data.Data(X, Y)

    kern = ScaleKernel(RBF(lengthscales=[rbf_ls]), rbf_var)
    sparsity = stgp.sparsity.NoSparsity(Z = X)
    latent_gps = [GP(sparsity=sparsity, kernel=kern)]

    Q = 1

    m = GP(
        data = data,
        prior = Independent(latent_gps),
        likelihood = ProductLikelihood([Gaussian(lik_var)]),
        approximate_posterior = MeanFieldConjugateGaussian([
            ConjugateGaussian(
                X=sparsity,
                block_size=1,
                surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
                    data=Data(X.X, Y), # Data should already be in the correct format
                    prior=Independent([latent_gps[q]]), 
                    likelihood=[likelihood[q]]
                )  
            )
            for q in range(Q)
        ]),
        inference='Variational'
    )

    return m

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [50])
@pytest.mark.parametrize('NS', [100])
@pytest.mark.parametrize('lik_var', [0.2])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
def test__vgp_and_cvi_match_with_same_init(regression_1d_data, seed, N, NS, rbf_ls, rbf_var, lik_var, cvi_gp):
    # ==== Arrange ====
    X, Y = regression_1d_data

    mu_post, var_post = cvi_gp.approximate_posterior.approx_posteriors[0].surrogate.posterior(diagonal=False)

    # ==== Act ====

    # Construct VGP using mu_post, var_post
    data = stgp.data.Data(X, Y)
    m_vgp = stgp.models.GP(
        data = data, 
        kernel=ScaleKernel(RBF(lengthscales=[rbf_ls]), rbf_var),
        likelihood = [Gaussian(variance=lik_var)],
        inference='Variational',
        approximate_posterior = MeanFieldApproximatePosterior(
            approximate_posteriors=[
                GaussianApproximatePosterior(
                    m = mu_post[:, None],
                    S = var_post
                )
            ]
        )
    )
    vgp_elbo = m_vgp.get_objective()
    cvi_elbo = cvi_gp.get_objective()

    vgp_mu, vgp_var = m_vgp.predict_f(X)
    cvi_mu, cvi_var = cvi_gp.predict_f(X)

    # === Assert  ===
    np.testing.assert_allclose(vgp_elbo, cvi_elbo)
    np.testing.assert_allclose(vgp_mu, cvi_mu, rtol=1e-5)
    np.testing.assert_allclose(vgp_var, cvi_var,rtol=1e-3)
