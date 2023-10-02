""" Unittests for the input arguments for state-space models """
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
from ..common_fixtures import regression_2d_data, regression_1d_data, regression_1d_diff_obs_data, regression_2d_diff_obs_data, multi_output_timeseries
from stgp.models import GP
from stgp.kernels import Matern32, SpatioTemporalSeperableKernel, ScaledMatern32, ScaleKernel
from stgp.data import Data, SpatioTemporalData, TemporalData, MultiOutputTemporalData
from stgp.likelihood import Gaussian, ReshapedGaussian, DiagonalGaussian, GaussianProductLikelihood
from stgp.transforms.sdes import LTI_SDE, LTI_SDE_Full_State_Obs
from stgp.transforms import Independent
from stgp.kernels.diff_op import FirstOrderDerivativeKernel
from stgp.transforms.pdes import DifferentialOperatorJoint

def get_1d_sde_gp(X, Y, seed):
    np.random.seed(seed)
    stgp.settings.jitter = 1e-7

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

def get_1d_multi_latent_sde_gp(X, Y, seed):
    np.random.seed(seed)
    stgp.settings.jitter = 1e-7

    Q = Y.shape[1]

    # Construct Model
    data = MultiOutputTemporalData(X, Y)

    lik = ReshapedGaussian(
        GaussianProductLikelihood([Gaussian(variance=1.0) for q in range(Q)]),
        num_blocks=data.Nt, 
        block_size=Q
    )
    prior = LTI_SDE(Independent(
        [
            GP(
                sparsity = stgp.sparsity.NoSparsity(Z_ref = data.X), 
                kernel = ScaledMatern32(input_dim=1, lengthscales=[1.0], variance=1.0),
                prior = True
            )
            for q in range(Q)
        ]
    )) 


    m = GP(data = data, prior = prior, likelihood = lik, inference='Sequential')

    return m


@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('NS', [200])
def test__1d_sde_gp_predict_f__full_state(seed, N, NS, regression_1d_data):
    # ==== Arrange ====
    X, Y = regression_1d_data
    
    #construct testing data
    XS = np.linspace(0, 1, NS)[:, None]

    # get model
    state_dim = 2 # matern32 kernel is used
    sde_gp = get_1d_sde_gp(X, Y, seed)

    # ==== Act ====

    pred_mu, pred_var = sde_gp.predict_f(XS, force_full_state=True, diagonal=False, squeeze=False)

    # ==== Assert ====
    np.testing.assert_equal([NS, state_dim, 1], pred_mu.shape)
    np.testing.assert_equal([NS, 1, state_dim, state_dim], pred_var.shape)

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('NS', [200])
def test__1d_sde_gp_predict_f__full_state__diagonal(seed, N, NS, regression_1d_data):
    # ==== Arrange ====
    X, Y = regression_1d_data
    
    #construct testing data
    XS = np.linspace(0, 1, NS)[:, None]

    # get model
    state_dim = 2 # matern32 kernel is used
    sde_gp = get_1d_sde_gp(X, Y, seed)

    # ==== Act ====

    pred_mu, pred_var = sde_gp.predict_f(XS, force_full_state=True, diagonal=True, squeeze=False)

    # ==== Assert ====
    np.testing.assert_equal([NS, state_dim, 1], pred_mu.shape)
    np.testing.assert_equal([NS, state_dim, 1, 1], pred_var.shape)


@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('NS', [200])
def test__1d_sde_gp_predict_f(seed, N, NS, regression_1d_data):
    # ==== Arrange ====
    X, Y = regression_1d_data
    
    #construct testing data
    XS = np.linspace(0, 1, NS)[:, None]

    # get model
    state_dim = 2 # matern32 kernel is used
    sde_gp = get_1d_sde_gp(X, Y, seed)

    # ==== Act ====

    pred_mu, pred_var = sde_gp.predict_f(XS, force_full_state=False, diagonal=False, squeeze=False)

    # ==== Assert ====
    np.testing.assert_equal([NS, 1, 1], pred_mu.shape)
    np.testing.assert_equal([NS, 1, 1, 1], pred_var.shape)

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('NS', [200])
def test__1d_sde_gp_predict_f__diagonal(seed, N, NS, regression_1d_data):
    # ==== Arrange ====
    X, Y = regression_1d_data
    
    #construct testing data
    XS = np.linspace(0, 1, NS)[:, None]

    # get model
    state_dim = 2 # matern32 kernel is used
    sde_gp = get_1d_sde_gp(X, Y, seed)

    # ==== Act ====

    pred_mu, pred_var = sde_gp.predict_f(XS, force_full_state=False, diagonal=True, squeeze=False)

    # ==== Assert ====
    np.testing.assert_equal([NS, 1, 1], pred_mu.shape)
    np.testing.assert_equal([NS, 1, 1, 1], pred_var.shape)


@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('NS', [200])
def test__1d_sde_gp_predict_f__filter_only(seed, N, NS, regression_1d_data):
    # ==== Arrange ====
    X, Y = regression_1d_data
    
    #construct testing data
    XS = np.linspace(0, 1, NS)[:, None]

    # get model
    state_dim = 2 # matern32 kernel is used
    sde_gp = get_1d_sde_gp(X, Y, seed)

    # ==== Act ====

    pred_mu, pred_var = sde_gp.predict_f(XS, filter_only=True, force_full_state=False, diagonal=False, squeeze=False)

    # ==== Assert ====
    np.testing.assert_equal([NS, state_dim, 1], pred_mu.shape)
    np.testing.assert_equal([NS, 1, state_dim, state_dim], pred_var.shape)

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('NS', [200])
def test__1d_sde_gp_predict_f__filter_only_diagonal(seed, N, NS, regression_1d_data):
    # ==== Arrange ====
    X, Y = regression_1d_data
    
    #construct testing data
    XS = np.linspace(0, 1, NS)[:, None]

    # get model
    state_dim = 2 # matern32 kernel is used
    sde_gp = get_1d_sde_gp(X, Y, seed)

    # ==== Act ====

    pred_mu, pred_var = sde_gp.predict_f(XS, filter_only=True, force_full_state=False, diagonal=True, squeeze=False)

    # ==== Assert ====
    np.testing.assert_equal([NS, state_dim, 1], pred_mu.shape)
    np.testing.assert_equal([NS, state_dim, 1, 1], pred_var.shape)

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('NS', [200])
@pytest.mark.parametrize('P', [3])
def test__1d_multi_latent_sde_gp_predict_f(seed, N, NS, multi_output_timeseries):
    # ==== Arrange ====
    X, Y = multi_output_timeseries
    
    #construct testing data
    XS = np.linspace(0, 1, NS)[:, None]

    # get model
    out_dim = 3
    state_dim = 2*out_dim # matern32 kernel is used and there are three outputs
    sde_gp = get_1d_multi_latent_sde_gp(X, Y, seed)

    # ==== Act ====
    pred_mu, pred_var = sde_gp.predict_f(XS, filter_only=False, force_full_state=False, diagonal=False, squeeze=False)

    # ==== Assert ====
    np.testing.assert_equal([NS, out_dim, 1], pred_mu.shape)
    np.testing.assert_equal([NS, 1, out_dim, out_dim], pred_var.shape)

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('NS', [200])
@pytest.mark.parametrize('P', [3])
def test__1d_multi_latent_sde_gp_predict_f_diagonal(seed, N, NS, multi_output_timeseries):
    # ==== Arrange ====
    X, Y = multi_output_timeseries
    
    #construct testing data
    XS = np.linspace(0, 1, NS)[:, None]

    # get model
    out_dim = 3
    state_dim = 2*out_dim # matern32 kernel is used and there are three outputs
    sde_gp = get_1d_multi_latent_sde_gp(X, Y, seed)

    # ==== Act ====
    pred_mu, pred_var = sde_gp.predict_f(XS, filter_only=False, force_full_state=False, diagonal=True, squeeze=False)

    # ==== Assert ====
    np.testing.assert_equal([NS, out_dim, 1], pred_mu.shape)
    np.testing.assert_equal([NS, out_dim, 1, 1], pred_var.shape)

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('NS', [200])
@pytest.mark.parametrize('P', [3])
def test__1d_multi_latent_sde_gp_predict_f__full_state(seed, N, NS, multi_output_timeseries):
    # ==== Arrange ====
    X, Y = multi_output_timeseries
    
    #construct testing data
    XS = np.linspace(0, 1, NS)[:, None]

    # get model
    out_dim = 3
    state_dim = 2*out_dim # matern32 kernel is used and there are three outputs
    sde_gp = get_1d_multi_latent_sde_gp(X, Y, seed)

    # ==== Act ====
    pred_mu, pred_var = sde_gp.predict_f(XS, filter_only=False, force_full_state=True, diagonal=False, squeeze=False)

    # ==== Assert ====
    np.testing.assert_equal([NS, state_dim, 1], pred_mu.shape)
    np.testing.assert_equal([NS, 1, state_dim, state_dim], pred_var.shape)

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('NS', [200])
@pytest.mark.parametrize('P', [3])
def test__1d_multi_latent_sde_gp_predict_f__full_state_diagonal(seed, N, NS, multi_output_timeseries):
    # ==== Arrange ====
    X, Y = multi_output_timeseries
    
    #construct testing data
    XS = np.linspace(0, 1, NS)[:, None]

    # get model
    out_dim = 3
    state_dim = 2*out_dim # matern32 kernel is used and there are three outputs
    sde_gp = get_1d_multi_latent_sde_gp(X, Y, seed)

    # ==== Act ====
    pred_mu, pred_var = sde_gp.predict_f(XS, filter_only=False, force_full_state=True, diagonal=True, squeeze=False)

    # ==== Assert ====
    np.testing.assert_equal([NS, state_dim, 1], pred_mu.shape)
    np.testing.assert_equal([NS, state_dim, 1, 1], pred_var.shape)


