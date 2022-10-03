""" Unittests for dispatched full-posterior marginals. """
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

from ..common_fixtures import regression_1d_data, rbf_1d_kernel, gaussian_likelihood, full_posterior_joint_model_no_sparsity, gp_prior_1d, gaussian_likelihood, gp_model_1d


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



@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [50])
@pytest.mark.parametrize('NS', [100])
def test__non_zero_gradients_vgp__no_whitening(seed, N, NS, regression_1d_data):
    # ==== Arrange ====
    X, Y = regression_1d_data
    data = stgp.data.Data(X, Y)

    m = stgp.models.GP(
        data = data, 
        kernel=ScaleKernel(RBF(lengthscales=[1.0])),
        likelihood = Gaussian(variance=1.0),
        inference='Variational'
    )

    # ==== Act ====
    obj_fn = objax.Jit(m.get_objective, m.vars())
    grad_fn = objax.Grad(obj_fn, m.vars())

    # ==== Assert ====

    assert np.all(np.array([np.sum(np.abs(g)) for g in grad_fn()]) > 0)

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [50])
@pytest.mark.parametrize('NS', [100])
def test__non_zero_gradients_vgp__with_whitening(seed, N, NS, regression_1d_data):
    # ==== Arrange ====
    X, Y = regression_1d_data
    data = stgp.data.Data(X, Y)

    m = stgp.models.GP(
        data = data, 
        kernel=ScaleKernel(RBF(lengthscales=[1.0])),
        likelihood = Gaussian(variance=1.0),
        inference='Variational',
        whiten=True
    )

    # ==== Act ====
    obj_fn = objax.Jit(m.get_objective, m.vars())
    grad_fn = objax.Grad(obj_fn, m.vars())

    # ==== Assert ====

    assert np.all(np.array([np.sum(np.abs(g)) for g in grad_fn()]) > 0)
