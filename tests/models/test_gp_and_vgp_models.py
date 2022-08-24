""" Unittests for dispatched full-posterior marginals. """
import jax 
import jax.numpy as jnp
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', True)

import pytest
import numpy as np
import scipy

import stgp
from stgp import settings
from stgp.models import GP
from stgp.computation.elbos.kullback_leiblers import gaussian_cholesky_kl, gaussian_kl, whitened_gaussian_kl
from stgp.dispatch import evoke
from stgp.sparsity import NoSparsity
from stgp.transforms.pdes import DifferentialOperatorJoint
from stgp.approximate_posteriors import FullGaussianApproximatePosterior, MeanFieldApproximatePosterior
from stgp.kernels import RBF, ScaleKernel
from stgp.likelihood import Gaussian
from stgp.kernels.diff_op import SecondOrderDerivativeKernel_2D
from stgp.trainers import NatGradTrainer

from ..common_fixtures import regression_1d_data, rbf_1d_kernel, gaussian_likelihood, full_posterior_joint_model_no_sparsity, gp_prior_1d, gaussian_likelihood, gp_model_1d

@pytest.fixture
def vgp(regression_1d_data, lik_var, rbf_ls, rbf_var):
    X, Y = regression_1d_data
    data = stgp.data.Data(X, Y)


    # Create Model
    m = stgp.models.GP(
        data = data, 
        kernel=ScaleKernel(RBF(lengthscales=[rbf_ls]), rbf_var),
        likelihood = [Gaussian(variance=lik_var)],
        inference='Variational'
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

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [50])
@pytest.mark.parametrize('NS', [100])
@pytest.mark.parametrize('lik_var', [0.2])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
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
@pytest.mark.parametrize('N', [50])
@pytest.mark.parametrize('NS', [100])
def test__ways_to_construct_vgp_match(seed, N, NS, regression_1d_data):
    # ==== Arrange ====
    X, Y = regression_1d_data
    data = stgp.data.Data(X, Y)

    # ==== Act ====

    def implicit_method():
        m = GP(data=data, inference='Variational')
        return m

    def implicit_meanfield_method():
        m = stgp.models.GP(
            data = data, 
            kernel=RBF(lengthscales=[1.0]),
            likelihood = Gaussian(variance=1.0),
            inference='Variational'
        )
        return m

    def explicit_meanfield_method():
        m = stgp.models.GP(
            data = data, 
            kernel=RBF(lengthscales=[1.0]),
            likelihood = [Gaussian(variance=1.0)],
            inference='Variational',
            approximate_posterior = MeanFieldApproximatePosterior(dim_list=[X.shape[0]])
        )
        return m

    def explicit_prior_meanfield_method():
        ind_prior = stgp.transforms.Independent([
            stgp.models.GP(
                sparsity = stgp.sparsity.NoSparsity(X),
                kernel = RBF(lengthscales=[1.0]),
                prior = True
            )
        ])
        m = stgp.models.GP(
            data = data, 
            likelihood = [Gaussian(variance=1.0)],
            inference='Variational',
            approximate_posterior = MeanFieldApproximatePosterior(dim_list=[X.shape[0]])
        )
        return m

    def implicit_ind_prior_meanfield_method():
        ind_prior = stgp.models.GP(
            sparsity = stgp.sparsity.NoSparsity(X),
            kernel = RBF(lengthscales=[1.0]),
            prior = True
        )

        m = stgp.models.GP(
            data = data, 
            likelihood = Gaussian(variance=1.0),
            inference='Variational'
        )
        return m

    def explicit_full_posterior_method():
        m = stgp.models.GP(
            data = data, 
            kernel=RBF(lengthscales=[1.0]),
            likelihood = [Gaussian(variance=1.0)],
            inference='Variational',
            approximate_posterior = FullGaussianApproximatePosterior(dim = X.shape[0])
        )
        return m

    m_implicit = implicit_method()
    m_implicit_mf = implicit_meanfield_method()
    m_explicit_meanfield_method = explicit_meanfield_method()
    m_explicit_prior_meanfield_method = explicit_prior_meanfield_method()
    m_implicit_ind_prior_meanfield_method = implicit_ind_prior_meanfield_method()
    m_explicit_full_posterior_method = explicit_full_posterior_method()

    base_model = m_implicit

    # === Assert  ===

    base_model_obj = base_model.get_objective()
    base_pred_mu, base_pred_var = base_model.predict_y(X)

    def assert_equal(m):
        m_pred_mu, m_pred_var = m.predict_y(X)

        np.testing.assert_allclose(
            base_model_obj, 
            m.get_objective()
        )

        np.testing.assert_allclose(
            base_pred_mu, 
            m_pred_mu
        )
        np.testing.assert_allclose(
            base_pred_var, 
            m_pred_var
        )

    assert_equal(m_implicit_mf)
    assert_equal(m_explicit_meanfield_method)
    assert_equal(m_explicit_prior_meanfield_method)
    assert_equal(m_implicit_ind_prior_meanfield_method)
    assert_equal(m_explicit_full_posterior_method)
