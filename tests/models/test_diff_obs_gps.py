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
from stgp.likelihood import Gaussian, ReshapedGaussian, DiagonalGaussian, BlockDiagonalGaussian, ReshapedDiagonalGaussian, ProductLikelihood
from stgp.transforms.sdes import LTI_SDE, LTI_SDE_Full_State_Obs
from stgp.transforms import Independent
from stgp.kernels.diff_op import FirstOrderDerivativeKernel
from stgp.transforms.pdes import DifferentialOperatorJoint
from stgp.means.mean import FirstOrderDerivativeMean
from stgp.approximate_posteriors import FullGaussianApproximatePosterior, MeanFieldApproximatePosterior, MeanFieldConjugateGaussian, FullConjugateGaussian
from stgp.trainers import NatGradTrainer

from stgp.zoo.sde_diff import diff_sparse_sde_vgp
from stgp.zoo.diff import diff_gp, diff_vgp, diff_hierarchical_sde_vgp, diff_hierarchical_sparse_sde_vgp, diff_sde_vgp, diff_hierarchical_vgp

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [20])
@pytest.mark.parametrize('NS', [100])
@pytest.mark.parametrize('kernel_ls', [0.1])
@pytest.mark.parametrize('kernel_var', [0.6])
@pytest.mark.parametrize('lik_var', [[0.05, 1.3]])
def test__1d_batch_gps_with_temporal_diff_obs_match(seed, N, NS, regression_1d_diff_obs_data, kernel_ls, kernel_var, lik_var):
    """ Check that a BatchGP with a FirstOrder kernel matches an explicitely created kernel like FirstOrder"""
    # ==== Arrange ====
    X, Y = regression_1d_diff_obs_data

    XS = np.linspace(np.min(X), np.max(X), NS)[:, None]

    # shuffle the rows of XS
    np.random.seed(seed)
    np.random.shuffle(XS)

    stgp.settings.jitter = 1e-7

    # ==== Act ====

    def batch_model():
        base_kernel = ScaleKernel(
            Matern32(input_dim = 1, lengthscales = [kernel_ls], active_dims=[0])  
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

    def vgp_model():

        base_kernel = ScaleKernel(
            Matern32(input_dim = 1, lengthscales = [kernel_ls], active_dims=[0])  
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

        q = FullGaussianApproximatePosterior(dim = X.shape[0] * diff_op_prior.output_dim)


        # Create Model
        m = stgp.models.GP(
            data = stgp.data.Data(X, Y),
            prior = diff_op_prior,
            likelihood = [Gaussian(lik_var[0]), Gaussian(lik_var[1])],
            inference='Variational',
            approximate_posterior = q
        )

        return m

    def sde_model():
        base_kernel =ScaledMatern32(input_dim = 1, lengthscales = [kernel_ls], variance=kernel_var)

        latent_gp = GP(
            sparsity=stgp.sparsity.NoSparsity(Z=X), 
            kernel = base_kernel
        )
        latent_gp = LTI_SDE_Full_State_Obs(Independent([latent_gp]))

        data = stgp.data.MultiOutputTemporalData(X=X, Y=Y, sort=True)
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

    def sde_vgp_model():
        time_kernel = ScaledMatern32(input_dim = 1, lengthscales = [kernel_ls], variance=kernel_var, active_dims=[0])
        m = diff_sde_vgp(X, Y, time_diff = 1, space_diff = None, time_kernel = time_kernel, space_kernel = None, lik_var = lik_var, fix_y = True)
   
        return m

    m_batch = batch_model()
    m_sde = sde_model()
    m_sde_vgp = sde_vgp_model()
    m_vgp = vgp_model()

    # vgp will match after a single nat grad step
    NatGradTrainer(m_vgp).train(1.0, 1)
    NatGradTrainer(m_sde_vgp).train(1.0, 1)


    # ==== Assert ====
    batch_pred, batch_var = m_batch.predict_f(XS)
    sde_pred, sde_var = m_sde.predict_f(XS)
    vgp_pred, vgp_var = m_vgp.predict_f(XS)
    sde_vgp_pred, sde_vgp_var = m_sde_vgp.predict_f(XS)

    np.testing.assert_allclose(m_batch.get_objective(), m_sde.get_objective(), rtol=1e-5)
    np.testing.assert_allclose(m_batch.get_objective(), m_vgp.get_objective(), rtol=1e-5)
    np.testing.assert_allclose(m_batch.get_objective(), m_sde_vgp.get_objective(), rtol=1e-5)


    # if this passes then it is likely that the predictions are correct
    #   but in the wrong order
    np.testing.assert_allclose(np.sum(batch_pred), np.sum(sde_pred), rtol=1e-5)
    np.testing.assert_allclose(np.sum(batch_pred), np.sum(vgp_pred), rtol=1e-3)
    np.testing.assert_allclose(np.sum(batch_pred), np.sum(sde_vgp_pred), rtol=1e-3)

    np.testing.assert_allclose(np.sum(batch_var), np.sum(sde_var), rtol=1e-5)
    np.testing.assert_allclose(np.sum(batch_var), np.sum(vgp_var), rtol=1e-3)
    np.testing.assert_allclose(np.sum(batch_var), np.sum(sde_vgp_var), rtol=1e-3)

    # now test that all the individual predictions match
    np.testing.assert_allclose(np.squeeze(batch_pred), np.squeeze(sde_pred), rtol=1e-3)
    np.testing.assert_allclose(np.squeeze(batch_var), np.squeeze(sde_var), rtol=1e-3)
    # use a slightly smaller rtol for variational methods to accounts for small
    #  computational differences 
    np.testing.assert_allclose(np.squeeze(batch_pred), np.squeeze(vgp_pred), atol=1e-3)
    np.testing.assert_allclose(np.squeeze(batch_var), np.squeeze(vgp_var), atol=1e-3)

    # use a slightly smaller rtol for variational methods to accounts for small
    #  computational differences 
    np.testing.assert_allclose(np.squeeze(batch_pred), np.squeeze(sde_vgp_pred), atol=1e-3)
    np.testing.assert_allclose(np.squeeze(batch_var), np.squeeze(sde_vgp_var), atol=1e-3)

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

    def vgp_model():

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

        q = FullGaussianApproximatePosterior(dim = X.shape[0] * diff_op_prior.output_dim)


        # Create Model
        m = stgp.models.GP(
            data = stgp.data.Data(X, Y),
            prior = diff_op_prior,
            likelihood = [Gaussian(lik_var[0]), Gaussian(lik_var[1])],
            inference='Variational',
            approximate_posterior = q
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

    # vgp will match after a single nat grad step
    m_vgp = vgp_model()
    NatGradTrainer(m_vgp).train(1.0, 1)



    # ==== Assert ====
    batch_pred, batch_var = m_batch.predict_f(XS)
    sde_pred, sde_var = m_sde.predict_f(XS)
    vgp_pred, vgp_var = m_vgp.predict_f(XS)

    # only keep diagonal elements
    sde_var = np.diagonal(sde_var, axis1=2, axis2=3)[:, 0, ...]

    np.testing.assert_allclose(m_batch.get_objective(), m_sde.get_objective(), rtol=1e-5)
    np.testing.assert_allclose(m_batch.get_objective(), m_vgp.get_objective(), rtol=1e-5)


    # if this passes then it is likely that the predictions are correct
    #   but in the wrong order
    np.testing.assert_allclose(np.sum(batch_pred), np.sum(sde_pred), rtol=1e-5)
    np.testing.assert_allclose(np.sum(batch_pred), np.sum(vgp_pred), rtol=1e-3)

    np.testing.assert_allclose(np.sum(batch_var), np.sum(sde_var), rtol=1e-5)
    np.testing.assert_allclose(np.sum(batch_var), np.sum(vgp_var), rtol=1e-3)

    # now test that all the individual predictions match
    np.testing.assert_allclose(np.squeeze(batch_pred), np.squeeze(sde_pred), rtol=1e-3)
    np.testing.assert_allclose(np.squeeze(batch_var), np.squeeze(sde_var), rtol=1e-3)
    # use a slightly smaller rtol for variational methods to accounts for small
    #  computational differences 
    np.testing.assert_allclose(np.squeeze(batch_pred), np.squeeze(vgp_pred), atol=1e-3)
    np.testing.assert_allclose(np.squeeze(batch_var), np.squeeze(vgp_var), atol=1e-3)


@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('NS', [15])
@pytest.mark.parametrize('kernel_ls', [0.1])
@pytest.mark.parametrize('kernel_var', [0.6])
@pytest.mark.parametrize('lik_var', [[0.05, 1.3, 0.1, 0.3]])
def test__2d_batch_gps_with_spatial_diff_obs_match(seed, N, NS, regression_2d_diff_obs_data, kernel_ls, kernel_var, lik_var):
    """ Check that a BatchGP with a FirstOrder2D kernel matches an explicitely created kernel like FirstOrder[FirstOrder]"""
    # ==== Arrange ====
    X, Y = regression_2d_diff_obs_data

    # Y is organised as [f, ds, dt, dtds] which is dt-ds format
   
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

        # only compute derivate kernel on the spatial dimension (axis=1)
        time_kern = FirstOrderDerivativeKernel(base_kernel, input_index=0)
        kern = FirstOrderDerivativeKernel(time_kern, input_index=1, parent_output_dim=2)

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
            likelihood = [Gaussian(lik_var[0]), Gaussian(lik_var[1]), Gaussian(lik_var[2]), Gaussian(lik_var[3])],
        )

        return m

    def vgp_model():

        base_kernel = ScaleKernel(
            Matern32(input_dim = 1, lengthscales = [kernel_ls], active_dims=[0]) * 
            Matern32(input_dim = 1, lengthscales = [kernel_ls], active_dims=[1])
        , kernel_var)

        # only compute derivate kernel on the temporal dimension (axis=0)
        time_kern = FirstOrderDerivativeKernel(base_kernel, input_index=0)
        kern = FirstOrderDerivativeKernel(time_kern, input_index=1, parent_output_dim=2)

        diff_op_prior = DifferentialOperatorJoint(
            GP(
                sparsity=stgp.sparsity.NoSparsity(Z=X), 
                kernel = base_kernel
            ),
            kernel = kern,
            is_base = True,
            has_parent=False
        )

        q = FullGaussianApproximatePosterior(dim = X.shape[0] * diff_op_prior.output_dim)

        # Create Model
        m = stgp.models.GP(
            data = stgp.data.Data(X, Y),
            prior = diff_op_prior,
            likelihood = [Gaussian(lik_var[0]), Gaussian(lik_var[1]), Gaussian(lik_var[2]), Gaussian(lik_var[3])],
            inference='Variational',
            approximate_posterior = q
        )

        return m

    def sde_model():
        base_kernel = SpatioTemporalSeperableKernel(
            FirstOrderDerivativeKernel(ScaledMatern32(input_dim = 1, lengthscales = [kernel_ls], variance=kernel_var), input_index=0),
            FirstOrderDerivativeKernel(Matern32(input_dim=1, lengthscales=[0.1], active_dims=[1]), input_index=1),
            spatial_output_dim = 2
        )
        latent_gp = GP(
            sparsity=stgp.sparsity.NoSparsity(Z=X), 
            kernel = base_kernel
        )
        latent_gp = LTI_SDE_Full_State_Obs(Independent([latent_gp]))

        data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)

        # state = f, df/ds, df/dt, ddf/dsdt
        Q = 4

        var = 0.1 * np.tile(np.eye(Q * data.Ns), [data.Nt, 1, 1]) 
        # block diagonal likelihood
        lik = ReshapedDiagonalGaussian(
            DiagonalGaussian([lik_var[0], lik_var[1], lik_var[2], lik_var[3]]), 
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

    # vgp will match after a single nat grad step
    m_vgp = vgp_model()
    NatGradTrainer(m_vgp).train(1.0, 1)

    # ==== Assert ====
    batch_pred, batch_var = m_batch.predict_f(XS)
    sde_pred, sde_var = m_sde.predict_f(XS)
    vgp_pred, vgp_var = m_vgp.predict_f(XS)

    # only keep diagonal elements
    sde_var = np.diagonal(sde_var, axis1=2, axis2=3)[:, 0, ...]

    np.testing.assert_allclose(m_batch.get_objective(), m_sde.get_objective(), rtol=1e-5)
    np.testing.assert_allclose(m_batch.get_objective(), m_vgp.get_objective(), rtol=1e-5)

    # if this passes then it is likely that the predictions are correct
    #   but in the wrong order
    np.testing.assert_allclose(np.sum(batch_pred), np.sum(sde_pred), rtol=1e-5)
    np.testing.assert_allclose(np.sum(batch_pred), np.sum(vgp_pred), rtol=1e-3)

    np.testing.assert_allclose(np.sum(batch_var), np.sum(sde_var), rtol=1e-5)
    np.testing.assert_allclose(np.sum(batch_var), np.sum(vgp_var), rtol=1e-3)

    # now test that all the individual predictions match
    if False:
        np.testing.assert_allclose(np.squeeze(batch_pred), np.squeeze(sde_pred), rtol=1e-3)
        np.testing.assert_allclose(np.squeeze(batch_pred), np.squeeze(vgp_pred), rtol=1e-1)

        np.testing.assert_allclose(np.squeeze(batch_var), np.squeeze(sde_var), rtol=1e-3)
        np.testing.assert_allclose(np.squeeze(batch_var), np.squeeze(vgp_var), rtol=1e-1)


@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [20])
@pytest.mark.parametrize('NS', [15])
@pytest.mark.parametrize('kernel_ls', [0.1])
@pytest.mark.parametrize('kernel_var', [0.6])
@pytest.mark.parametrize('lik_var', [[0.05, 1.3, 0.1, 0.3]])
def test__2d_sde_vgp_gps_with_spatial_diff_obs_match(seed, N, NS, regression_2d_diff_obs_data, kernel_ls, kernel_var, lik_var):
    """ Check that a BatchGP with a FirstOrder2D kernel matches an explicitely created kernel like FirstOrder[FirstOrder]"""
    # ==== Arrange ====
    X, Y = regression_2d_diff_obs_data

    # Y is organised as [f, ds, dt, dtds] which is dt-ds format
   
    #construct testing data
    x1, x2 = np.meshgrid(np.linspace(0, 1, NS), np.linspace(0, 1, NS))
    XS = np.hstack([np.reshape(x1, [NS*NS, 1]), np.reshape(x2, [NS*NS, 1])])

    # shuffle the rows of XS
    np.random.seed(seed)
    np.random.shuffle(XS)

    stgp.settings.jitter = 1e-5
    stgp.settings.ng_jitter = 1e-5

    # ==== Act ====

    def batch_model():
        base_kernel = ScaleKernel(
            Matern32(input_dim = 1, lengthscales = [kernel_ls], active_dims=[0]) * 
            Matern32(input_dim = 1, lengthscales = [kernel_ls], active_dims=[1])
        , kernel_var)

        # only compute derivate kernel on the spatial dimension (axis=1)
        time_kern = FirstOrderDerivativeKernel(base_kernel, input_index=0)
        kern = FirstOrderDerivativeKernel(time_kern, input_index=1, parent_output_dim=2)

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
            likelihood = [Gaussian(lik_var[0]), Gaussian(lik_var[1]), Gaussian(lik_var[2]), Gaussian(lik_var[3])],
        )

        return m

    def sde_vgp_model():
        time_kernel = ScaledMatern32(input_dim = 1, lengthscales = [kernel_ls], variance=kernel_var, active_dims=[0])
        space_kernel = Matern32(input_dim=1, lengthscales=[0.1], active_dims=[1])

        m = diff_sde_vgp(X, Y, time_diff = 1, space_diff = 1, time_kernel = time_kernel, space_kernel = space_kernel, lik_var = lik_var, fix_y = True)
   
        return m

    def sparse_sde_vgp_model():
        time_kernel = ScaledMatern32(input_dim = 1, lengthscales = [kernel_ls], variance=kernel_var, active_dims=[0])
        space_kernel = Matern32(input_dim=1, lengthscales=[0.1], active_dims=[1])

        data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)

        # no inducing points
        Z_s = data.X_space

        m = diff_sparse_sde_vgp(X, Y, time_diff = 1, space_diff = 1, time_kernel = time_kernel, space_kernel = space_kernel, lik_var = 0.1, fix_y = True, Z = Z_s)

        return m

    m_batch = batch_model()

    # vgp will match after a single nat grad step
    print('nat grad')
    m_sde_vgp = sde_vgp_model()
    NatGradTrainer(m_sde_vgp).train(1.0, 1)



    # ==== Assert ====
    print('predicting')
    batch_pred, batch_var = m_batch.predict_f(XS)
    vgp_pred, vgp_var = m_sde_vgp.predict_f(XS)

    if False:
        m_sparse_sde_vgp = sparse_sde_vgp_model()
        NatGradTrainer(m_sparse_sde_vgp).train(1.0, 1)
        sparse_vgp_pred, sparse_vgp_var = m_sparse_sde_vgp.predict_f(XS)

    #np.testing.assert_allclose(np.squeeze(batch_pred), np.squeeze(vgp_pred), rtol=1e-3)
    np.testing.assert_allclose(np.sum(batch_pred), np.sum(vgp_pred), rtol=1e-3)
    np.testing.assert_allclose(m_batch.get_objective(), m_sde_vgp.get_objective(), rtol=1e-5)
    #np.testing.assert_allclose(m_batch.get_objective(), m_sparse_sde_vgp.get_objective(), rtol=1e-5)
    #np.testing.assert_allclose(m_sde_vgp.get_objective(), m_sparse_sde_vgp.get_objective(), rtol=1e-5)

    # if this passes then it is likely that the predictions are correct
    #   but in the wrong order
    np.testing.assert_allclose(np.sum(batch_pred), np.sum(vgp_pred), rtol=1e-3)
    #np.testing.assert_allclose(np.sum(batch_pred), np.sum(sparse_vgp_pred), rtol=1e-3)

    np.testing.assert_allclose(np.sum(batch_var), np.sum(vgp_var), rtol=1e-3)
    #np.testing.assert_allclose(np.sum(batch_var), np.sum(sparse_vgp_var), rtol=1e-3)


@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [20])
@pytest.mark.parametrize('NS', [100])
@pytest.mark.parametrize('kernel_ls', [0.1])
@pytest.mark.parametrize('kernel_var', [0.6])
@pytest.mark.parametrize('lik_var', [[0.05, 1.3]])
def test__1d_multi_latent_recover_independent(seed, N, NS, regression_1d_diff_obs_data, kernel_ls, kernel_var, lik_var):
    # ==== Arrange ====
    X, Y1 = regression_1d_diff_obs_data
    Y2 = 5*np.copy(Y1)
    Y = np.hstack([Y1, Y2])


    XS = np.linspace(np.min(X), np.max(X), NS)[:, None]

    # shuffle the rows of XS
    np.random.seed(seed)
    np.random.shuffle(XS)

    stgp.settings.jitter = 1e-7

    def separate_batch_model(X, Y):
        base_kernel = ScaleKernel(
            Matern32(input_dim = 1, lengthscales = [kernel_ls], active_dims=[0])  
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

    def independent_batch_model(X, Y):
        base_kernel = ScaleKernel(
            Matern32(input_dim = 1, lengthscales = [kernel_ls], active_dims=[0])  
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

        prior = Independent([diff_op_prior, diff_op_prior])


        # Create Model
        m = stgp.models.GP(
            data = stgp.data.Data(X, Y),
            prior = prior,
            likelihood = [ProductLikelihood([Gaussian(lik_var[0]), Gaussian(lik_var[1])]), ProductLikelihood([Gaussian(lik_var[0]), Gaussian(lik_var[1])])],
        )

        return m

    def multi_latent_cvi_model(X, Y):
        Z = np.linspace(0, 1, 5)[:, None]
        P = 2
        #sparsity=stgp.sparsity.FullSparsity(Z=Z)
        sparsity=stgp.sparsity.NoSparsity(Z=X)
        base_kernel_1d = ScaledMatern32(input_dim = 1, lengthscales = [kernel_ls], variance=kernel_var)

        base_gp = GP(
            sparsity=sparsity, 
            kernel = base_kernel_1d
        )

        kern = FirstOrderDerivativeKernel(base_kernel_1d, parent_output_dim=base_gp.output_dim)

        prior = Independent([
            DifferentialOperatorJoint(
                base_gp,
                kernel = kern,
                is_base = True,
                has_parent=False,
                hierarchical = False
            )
            for p in range(P)
        ])

        st_data = stgp.data.MultiOutputTemporalData(X, Y)
        N = X.shape[0]
        q = MeanFieldConjugateGaussian(
            approximate_posteriors = [
                FullConjugateGaussian(
                    X = st_data._X,
                    num_latents =  2,
                    block_size= 2,
                    num_blocks = st_data.Nt,
                    surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
                        # in state-space format
                        data = stgp.data.MultiOutputTemporalData(X=X, Y=np.reshape(Y, [st_data.Nt, 2, 1]), sort=False), # we need gradients Y so set to be trainable
                        likelihood=likelihood, 
                        prior=LTI_SDE_Full_State_Obs(Independent([prior.parent[p].parent])),
                        inference='Sequential',
                        parallel=False
                    )
                )
                for p in range(P)
            ]
        )

        if False :
            likelihood_arr = [
                ProductLikelihood([Gaussian(lik_var[0])]), 
                ProductLikelihood([Gaussian(lik_var[1])]), 
                ProductLikelihood([Gaussian(lik_var[0])]), 
                ProductLikelihood([Gaussian(lik_var[1])]),
            ]
        elif True:
            likelihood_arr = [Gaussian(lik_var[0]), Gaussian(lik_var[1]), Gaussian(lik_var[0]), Gaussian(lik_var[1])]
        else:
            likelihood_arr = [
                ProductLikelihood([Gaussian(lik_var[0]), Gaussian(lik_var[1])]), 
                ProductLikelihood([Gaussian(lik_var[0]), Gaussian(lik_var[1])])
            ]

        # Create Model
        m = stgp.models.GP(
            data = st_data,
            prior = prior,
            likelihood =likelihood_arr,
            inference='Variational',
            approximate_posterior = q
        )

        return m

    m1_batch = separate_batch_model(X, Y1)
    m2_batch = separate_batch_model(X, Y2)
    m1_ind = independent_batch_model(X, Y)
    m_cvi = multi_latent_cvi_model(X, Y)

    NatGradTrainer(m_cvi).train(1.0, 1)

    # assert same log marginal likelihood
    np.testing.assert_allclose(
        np.sum(m_cvi.get_objective()), 
        np.sum(m1_batch.get_objective()+m2_batch.get_objective()), 
    rtol=1e-5)

    m1_batch_pred_mu, m1_batch_pred_var = m1_batch.predict_f(XS)
    m2_batch_pred_mu, m2_batch_pred_var = m2_batch.predict_f(XS)
    m_batch_pred_mu = np.hstack([np.squeeze(m1_batch_pred_mu), np.squeeze(m2_batch_pred_mu)])
    m_batch_pred_var = np.hstack([np.squeeze(m1_batch_pred_var), np.squeeze(m2_batch_pred_var)])

    m_cvi_pred_mu, m_cvi_pred_var = m_cvi.predict_f(XS)
    m_cvi_pred_mu = np.squeeze(m_cvi_pred_mu)
    m_cvi_pred_var = np.diagonal(m_cvi_pred_var, axis1=2, axis2=3)[:, 0, :]

    np.testing.assert_allclose( np.sum(m_cvi_pred_mu), np.sum(m_batch_pred_mu), rtol=1e-5)
    np.testing.assert_allclose( np.sum(m_batch_pred_var), np.sum(m_cvi_pred_var), rtol=1e-5)
