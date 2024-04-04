""" Unittests for state-space models """
import jax 
import jax.numpy as jnp
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import objax

import pytest
import numpy as np
import scipy

import stgp
from ..common_fixtures import regression_2d_data, regression_1d_data, regression_1d_diff_obs_data, regression_2d_diff_obs_data
from stgp.models import GP
from stgp.kernels import Matern32, SpatioTemporalSeperableKernel, ScaledMatern32, ScaleKernel
from stgp.data import Data, SpatioTemporalData, TemporalData
from stgp.likelihood import Gaussian, ReshapedGaussian, DiagonalGaussian
from stgp.transforms.sdes import LTI_SDE, LTI_SDE_Full_State_Obs
from stgp.transforms import Independent
from stgp.kernels.diff_op import FirstOrderDerivativeKernel
from stgp.transforms.pdes import DifferentialOperatorJoint


@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('NS', [200])
def test__1d_sde_gp_and_batch_gp_match__shuffled(seed, N, NS, regression_1d_data):
    """ Check that a 1D SDE-GP matches a Batch GP"""
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
    """ Check that a 2D SDE-GP matches a Batch GP"""
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


@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('NS', [200])
@pytest.mark.parametrize('kernel_ls', [0.1])
@pytest.mark.parametrize('kernel_var', [0.6])
@pytest.mark.parametrize('lik_var', [0.05])
def test__1d_sde_gp_and_batch_gp_match_with_diff_obs(seed, N, NS, regression_1d_diff_obs_data, kernel_ls, kernel_var, lik_var):
    """ Check that a 1D SDE-GP matches a Batch GP with Derivate Observations"""
    # ==== Arrange ====
    X, Y = regression_1d_diff_obs_data
    
    #construct testing data
    XS = np.linspace(0, 1, NS)[:, None]

    # shuffle the rows of XS
    np.random.seed(seed)
    np.random.shuffle(XS)

    stgp.settings.jitter = 1e-7

    # ==== Act ====

    def batch_model():
        base_kernel_1d = ScaleKernel(Matern32(input_dim = 1, lengthscales = [kernel_ls]), kernel_var)
        kern = FirstOrderDerivativeKernel(base_kernel_1d)

        diff_op_prior = DifferentialOperatorJoint(
            GP(
                sparsity=stgp.sparsity.NoSparsity(Z=X), 
                kernel = base_kernel_1d
            ),
            kernel = kern,
            is_base = True,
            has_parent=False
        )


        # Create Model
        m = stgp.models.GP(
            data = stgp.data.Data(X, Y),
            prior = diff_op_prior,
            likelihood = [Gaussian(lik_var), Gaussian(lik_var)],
        )


        return m

    def sde_model():
        base_kernel_1d = ScaledMatern32(input_dim = 1, lengthscales = [kernel_ls], variance=kernel_var)
        latent_gp = GP(
            sparsity=stgp.sparsity.NoSparsity(Z=X), 
            kernel = base_kernel_1d
        )
        latent_gp = LTI_SDE_Full_State_Obs(Independent([latent_gp]))

        lik = ReshapedGaussian(DiagonalGaussian([lik_var, lik_var]), N, 2)

        # Create Model
        m = stgp.models.GP(
            data = stgp.data.MultiOutputTemporalData(X, Y, sort=True),
            prior = latent_gp,
            likelihood = lik,
            full_state_observed = True,
            inference='Sequential'
        )

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
@pytest.mark.parametrize('kernel_ls', [0.1])
@pytest.mark.parametrize('kernel_var', [0.6])
@pytest.mark.parametrize('lik_var', [0.05])
def test__1d_sde_gp_and_sde_gp_with_nan_diff_obs_match(seed, N, NS, regression_1d_diff_obs_data, kernel_ls, kernel_var, lik_var):
    """ Check that a 1D SDE-GP NaN Derivate Observations matches a (standard) SDE-GP"""
    # ==== Arrange ====
    X, Y = regression_1d_diff_obs_data
    
    #construct testing data
    XS = np.linspace(0, 1, NS)[:, None]

    # shuffle the rows of XS
    np.random.seed(seed)
    np.random.shuffle(XS)

    stgp.settings.jitter = 1e-7

    # ==== Act ====

    def sde_model_with_no_diff_obs():
        # remove diff obs
        data = TemporalData(X=X, Y=Y[:, 0][:, None], sort=True)

        lik = ReshapedGaussian(Gaussian(lik_var), num_blocks=data.Nt, block_size=data.Ns)

        kern = ScaledMatern32(input_dim=1, lengthscales=[kernel_ls], active_dims=[0], variance=kernel_var)

        latent_gp = GP(
            sparsity = stgp.sparsity.NoSparsity(Z_ref = data.X), 
            kernel = kern,
            prior = True
        )

        prior = LTI_SDE(Independent([latent_gp])) 

        m = GP(data = data, prior = prior, likelihood = lik, inference='Sequential')

        return m



    def sde_model_with_nan_obs():
        Y[:, 1] = np.NaN

        base_kernel_1d = ScaledMatern32(input_dim = 1, lengthscales = [kernel_ls], variance=kernel_var)
        latent_gp = GP(
            sparsity=stgp.sparsity.NoSparsity(Z=X), 
            kernel = base_kernel_1d
        )
        latent_gp = LTI_SDE_Full_State_Obs(Independent([latent_gp]))

        lik = ReshapedGaussian(DiagonalGaussian([lik_var, lik_var]), N, 2)

        # Create Model
        m = stgp.models.GP(
            data = stgp.data.MultiOutputTemporalData(X, Y, sort=True),
            prior = latent_gp,
            likelihood = lik,
            full_state_observed = True,
            inference='Sequential'
        )

        return m

    m_sde = sde_model_with_no_diff_obs()
    m_sde_nans = sde_model_with_nan_obs()

    # ==== Assert ====
    sde_nans_pred, sde_nans_var = m_sde_nans.predict_f(XS)
    sde_pred, sde_var = m_sde.predict_f(XS)

    # remove diff obs predictions
    sde_nans_pred = sde_nans_pred[:, 0, 0]
    sde_nans_var = sde_nans_var[:, 0, 0, 0]

    np.testing.assert_allclose(m_sde_nans.get_objective(), m_sde.get_objective())

    # if this passes then it is likely that the predictions are correct
    #   but in the wrong order
    np.testing.assert_allclose(np.sum(sde_nans_pred), np.sum(sde_pred), rtol=1e-5)
    np.testing.assert_allclose(np.sum(sde_nans_var), np.sum(sde_var), rtol=1e-5)

    np.testing.assert_allclose(np.squeeze(sde_nans_pred), np.squeeze(sde_pred), rtol=1e-3)
    np.testing.assert_allclose(np.squeeze(sde_nans_var), np.squeeze(sde_var), rtol=1e-3)


@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('NS', [200])
@pytest.mark.parametrize('kernel_ls', [0.1])
@pytest.mark.parametrize('kernel_var', [0.6])
@pytest.mark.parametrize('lik_var', [0.05])
def _test__2d_sde_gp_and_batch_gp_match_with_diff_obs(seed, N, NS, regression_2d_diff_obs_data, kernel_ls, kernel_var, lik_var):
    """ Check that a 2D SDE-GP matches a Batch GP with Derivate Observations"""
    # ==== Arrange ====
    X, Y = regression_2d_diff_obs_data
    # shuffle the rows of XS
    np.random.seed(seed)

    stgp.settings.jitter = 1e-7

    breakpoint()

    # ==== Act ====

    def batch_model():
        base_kernel_1d = ScaleKernel(Matern32(input_dim = 1, lengthscales = [kernel_ls]), kernel_var)
        kern = FirstOrderDerivativeKernel(base_kernel_1d)

        diff_op_prior = DifferentialOperatorJoint(
            GP(
                sparsity=stgp.sparsity.NoSparsity(Z=X), 
                kernel = base_kernel_1d
            ),
            kernel = kern,
            is_base = True,
            has_parent=False
        )


        # Create Model
        m = stgp.models.GP(
            data = stgp.data.Data(X, Y),
            prior = diff_op_prior,
            likelihood = [Gaussian(lik_var), Gaussian(lik_var)],
        )


        return m

    def sde_model():
        base_kernel_1d = ScaledMatern32(input_dim = 1, lengthscales = [kernel_ls], variance=kernel_var)
        latent_gp = GP(
            sparsity=stgp.sparsity.NoSparsity(Z=X), 
            kernel = base_kernel_1d
        )
        latent_gp = LTI_SDE_Full_State_Obs(Independent([latent_gp]))

        lik = ReshapedGaussian(DiagonalGaussian([lik_var, lik_var]), N, 2)

        # Create Model
        m = stgp.models.GP(
            data = stgp.data.MultiOutputTemporalData(X, Y, sort=True),
            prior = latent_gp,
            likelihood = lik,
            full_state_observed = True,
            inference='Sequential'
        )

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

