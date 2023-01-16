""" Unittests for GPs with derivative observations """
import jax 
import jax.numpy as jnp
from jax.config import config as jax_config
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
from stgp.likelihood import Gaussian, ReshapedGaussian, DiagonalGaussian, BlockDiagonalGaussian, ReshapedDiagonalGaussian
from stgp.transforms.sdes import LTI_SDE, LTI_SDE_Full_State_Obs
from stgp.transforms import Independent
from stgp.kernels.diff_op import FirstOrderDerivativeKernel
from stgp.transforms.pdes import DifferentialOperatorJoint

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('NS', [15])
@pytest.mark.parametrize('kernel_ls', [0.1])
@pytest.mark.parametrize('kernel_var', [0.6])
@pytest.mark.parametrize('lik_var', [[0.05, 1.3]])
def test__2d_batch_gps_with_temporal_diff_obs_match(seed, N, NS, regression_2d_diff_obs_data, kernel_ls, kernel_var, lik_var):
    """ Check that a BatchGP with a FirstOrder2D kernel matches an explicitely created kernel like FirstOrder[FirstOrder]"""
    # ==== Arrange ====
    X, Y = regression_2d_diff_obs_data

    # only select Y, dY/dt
    Y = np.hstack([Y[:, 0][:, None], Y[:, 1][:, None]])
    
    #construct testing data
    x1, x2 = np.meshgrid(np.linspace(0, 1, NS), np.linspace(0, 1, NS))
    XS = np.hstack([np.reshape(x1, [NS*NS, 1]), np.reshape(x2, [NS*NS, 1])])

    # shuffle the rows of XS
    np.random.seed(seed)
    np.random.shuffle(XS)

    stgp.settings.jitter = 1e-7

    # ==== Act ====

    def batch_model():
        base_kernel = ScaleKernel(
            Matern32(input_dim = 1, lengthscales = [kernel_ls], active_dims=[0]) * 
            Matern32(input_dim = 1, lengthscales = [kernel_ls], active_dims=[1])
        , kernel_var)

        # only compute derivate kernel on the temporal dimension (axis=0)
        kern = FirstOrderDerivativeKernel(base_kernel, input_index=0)

        diff_op_prior = DifferentialOperatorJoint(
            GP(
                sparsity=stgp.sparsity.NoSparsity(Z=X), 
                kernel = base_kernel
            ),
            kernel = kern,
            is_base = True,
            has_parent=False
        )


        # Create Model
        m = stgp.models.GP(
            data = stgp.data.Data(X, Y),
            prior = diff_op_prior,
            likelihood = [Gaussian(lik_var[0]), Gaussian(lik_var[1])],
        )

        return m

    def sde_model():
        base_kernel = SpatioTemporalSeperableKernel(
            FirstOrderDerivativeKernel(ScaledMatern32(input_dim = 1, lengthscales = [kernel_ls], variance=kernel_var), input_index=0),
            Matern32(input_dim=1, lengthscales=[0.1], active_dims=[1])
        )
        latent_gp = GP(
            sparsity=stgp.sparsity.NoSparsity(Z=X), 
            kernel = base_kernel
        )
        latent_gp = LTI_SDE_Full_State_Obs(Independent([latent_gp]))

        data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)
        Q = 2

        var = 0.1 * np.tile(np.eye(Q * data.Ns), [data.Nt, 1, 1]) 
        # block diagonal likelihood
        lik = ReshapedDiagonalGaussian(
            DiagonalGaussian([lik_var[0], lik_var[1]]), 
            data.Nt, 
            data.Ns,
            Q
        )

        # Create Model
        m = stgp.models.GP(
            data = data,
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
    # only keep diagonal elements
    sde_var = np.diagonal(sde_var, axis1=2, axis2=3)[:, 0, ...]

    np.testing.assert_allclose(m_batch.get_objective(), m_sde.get_objective(), rtol=1e-5)

    # if this passes then it is likely that the predictions are correct
    #   but in the wrong order
    np.testing.assert_allclose(np.sum(batch_pred), np.sum(sde_pred), rtol=1e-5)
    np.testing.assert_allclose(np.sum(batch_var), np.sum(sde_var), rtol=1e-5)

    np.testing.assert_allclose(np.squeeze(batch_pred), np.squeeze(sde_pred), rtol=1e-3)
    np.testing.assert_allclose(np.squeeze(batch_var), np.squeeze(sde_var), rtol=1e-3)

