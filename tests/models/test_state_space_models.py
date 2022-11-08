""" Unittests for state-space models """
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
from ..common_fixtures import regression_2d_data, regression_1d_data
from stgp.models import GP
from stgp.kernels import Matern32, SpatioTemporalSeperableKernel
from stgp.data import Data, SpatioTemporalData, TemporalData
from stgp.likelihood import Gaussian, ReshapedGaussian
from stgp.transforms.sdes import LTI_SDE
from stgp.transforms import Independent


@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('NS', [200])
def test__1d_sde_gp_and_batch_gp_match__shuffled(seed, N, NS, regression_1d_data):
    # ==== Arrange ====
    X, Y = regression_1d_data
    
    #construct testing data
    XS = np.linspace(0, 1, NS)[:, None]

    # shuffle the rows of XS
    np.random.seed(seed)
    np.random.shuffle(XS)

    stgp.settings.jitter = 1e-7

    # ==== Act ====

    def batch_model():
        data = Data(X, Y)
        kern = Matern32(input_dim=1, lengthscales=[0.1])
        m = GP(
            data = data,
            kernel = kern,
            likelihood = Gaussian(0.1)
        )

        return m

    def sde_model():
        data = TemporalData(X=X, Y=Y, sort=True)

        lik = ReshapedGaussian(Gaussian(0.1), num_blocks=data.Nt, block_size=data.Ns)

        kern = Matern32(input_dim=1, lengthscales=[0.1], active_dims=[0])

        latent_gp = GP(
            sparsity = stgp.sparsity.NoSparsity(Z_ref = data.X), 
            kernel = kern,
            prior = True
        )

        prior = LTI_SDE(Independent([latent_gp])) 

        m = GP(data = data, prior = prior, likelihood = lik, inference='Sequential')


        return m

    m_batch = batch_model()
    m_sde = sde_model()

    # ==== Assert ====
    batch_pred, batch_var = m_batch.predict_f(XS)
    sde_pred, sde_var = m_sde.predict_f(XS)

    np.testing.assert_allclose(m_batch.get_objective(), m_sde.get_objective())

    # if this passes then it is likely that the predictions are correct
    #   but in the wrong order
    np.testing.assert_allclose(np.sum(batch_pred), np.sum(sde_pred), rtol=1e-5)
    np.testing.assert_allclose(np.sum(batch_var), np.sum(sde_var), rtol=1e-5)

    np.testing.assert_allclose(np.squeeze(batch_pred), np.squeeze(sde_pred), rtol=1e-3)
    np.testing.assert_allclose(np.squeeze(batch_var), np.squeeze(sde_var), rtol=1e-3)


@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('NS', [200])
def test__2d_sde_gp_and_batch_gp_match__shuffled(seed, N, NS, regression_2d_data):
    # ==== Arrange ====
    X, Y = regression_2d_data
    
    #construct testing data
    x1, x2 = np.meshgrid(np.linspace(0, 1, NS), np.linspace(0, 1, NS))
    XS = np.hstack([np.reshape(x1, [NS*NS, 1]), np.reshape(x2, [NS*NS, 1])])

    # shuffle the rows of XS
    np.random.seed(seed)
    np.random.shuffle(XS)

    stgp.settings.jitter = 1e-7

    # ==== Act ====

    def batch_model():
        data = Data(X, Y)
        kern = Matern32(input_dim=2, lengthscales=[0.1, 0.1])
        m = GP(
            data = data,
            kernel = kern,
            likelihood = Gaussian(0.1)
        )

        return m

    def sde_model():
        data = SpatioTemporalData(X=X, Y=Y, sort=True)

        lik = ReshapedGaussian(Gaussian(0.1), num_blocks=data.Nt, block_size=data.Ns)

        kern = SpatioTemporalSeperableKernel(
            Matern32(input_dim=1, lengthscales=[0.1], active_dims=[0]),
            Matern32(input_dim=1, lengthscales=[0.1], active_dims=[1])
        )

        latent_gp = GP(
            sparsity = stgp.sparsity.NoSparsity(Z_ref = data.X), 
            kernel = kern,
            prior = True
        )

        prior = LTI_SDE(Independent([latent_gp])) 

        m = GP(data = data, prior = prior, likelihood = lik, inference='Sequential')


        return m

    m_batch = batch_model()
    m_sde = sde_model()

    # ==== Assert ====
    batch_pred, batch_var = m_batch.predict_f(XS)
    sde_pred, sde_var = m_sde.predict_f(XS)

    np.testing.assert_allclose(m_batch.get_objective(), m_sde.get_objective())

    # if this passes then it is likely that the predictions are correct
    #   but in the wrong order
    np.testing.assert_allclose(np.sum(batch_pred), np.sum(sde_pred), rtol=1e-5)
    np.testing.assert_allclose(np.sum(batch_var), np.sum(sde_var), rtol=1e-5)

    np.testing.assert_allclose(np.squeeze(batch_pred), np.squeeze(sde_pred), rtol=1e-3)
    np.testing.assert_allclose(np.squeeze(batch_var), np.squeeze(sde_var), rtol=1e-3)

