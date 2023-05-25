""" Unittests for Derivative kernels.  """

import jax 
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import pytest

import numpy as np
import scipy

import stgp
from stgp.kernels import RBF, ScaleKernel
from stgp.kernels.diff_op import SecondOrderDerivativeKernel_1D, FirstOrderDerivativeKernel, ClosedFormRBFFirstOrderDerivativeKernel


from ..common_fixtures import regression_1d_data, regression_2d_data

@pytest.mark.parametrize('N', [500])
@pytest.mark.parametrize('rbf_ls', [1.0, 0.1])
def test__ClosedFormFirstOrderDerivativeKernel(rbf_ls, N, regression_1d_data):
    # ==== Arrange ====
    X = np.linspace(0, 2, N)[:, None]

    # ==== Act ====
    # we construct an derivative kernel, take a sample and then assert 
    #   that the sampled first and second order matches the explictely
    #   computed gradients

    # get closed form kernel
    base_kernel_1d = RBF(input_dim = 1, lengthscales = [rbf_ls])

    # get true kernel use autograd
    kern_autograd = FirstOrderDerivativeKernel(base_kernel_1d, input_index=0)
    Kxx_true = kern_autograd.K(X, X)

    kern = ClosedFormRBFFirstOrderDerivativeKernel(base_kernel_1d, input_index=0)
    Kxx = kern.K(X, X)

    breakpoint()

    # check out samples
  
    # ==== assert ====
    # ignore the end points as the approximation from numpy is bad there
    # we have to a large a rtol due the approximation error of np.gradient
    np.testing.assert_allclose(Kxx_true, Kxx)


@pytest.mark.parametrize('N', [500])
@pytest.mark.parametrize('rbf_ls', [1.0, 0.1])
@pytest.mark.parametrize('rbf_var', [0.27])
def test__FirstOrderDerivativeKernel(rbf_ls, rbf_var, N, regression_1d_data):
    # ==== Arrange ====
    X = np.linspace(0, 2, N)[:, None]

    # ==== Act ====
    # we construct an derivative kernel, take a sample and then assert 
    #   that the sampled first and second order matches the explictely
    #   computed gradients

    base_kernel_1d = ScaleKernel(RBF(input_dim = 1, lengthscales = [rbf_ls]), rbf_var)

    kern = FirstOrderDerivativeKernel(base_kernel_1d, input_index=0)
    Kxx = kern.K(X, X)

    # check out samples
    latent = np.random.multivariate_normal(np.zeros(Kxx.shape[0]), Kxx)

    # there are three outputs
    latent_t = latent[:N]
    latent_dt = latent[N:N*2]

    # compute approximate derivatives
    approx_dt = np.gradient(latent_t, X[:, 0])

    # ==== assert ====
    # ignore the end points as the approximation from numpy is bad there
    # we have to a large a rtol due the approximation error of np.gradient
    np.testing.assert_allclose(approx_dt[2:-2], latent_dt[2:-2], atol=0.1)

@pytest.mark.parametrize('N', [500])
@pytest.mark.parametrize('rbf_ls', [1.0, 0.1])
@pytest.mark.parametrize('rbf_var', [0.27])
def test__SecondOrderDerivativeKernel_1D(rbf_ls, rbf_var, N, regression_1d_data):
    # ==== Arrange ====
    X = np.linspace(0, 2, N)[:, None]

    # ==== Act ====
    # we construct an derivative kernel, take a sample and then assert 
    #   that the sampled first and second order matches the explictely
    #   computed gradients

    base_kernel_1d = ScaleKernel(RBF(input_dim = 1, lengthscales = [rbf_ls]), rbf_var)

    kern = SecondOrderDerivativeKernel_1D(base_kernel_1d)
    Kxx = kern.K(X, X)

    # check out samples
    latent = np.random.multivariate_normal(np.zeros(Kxx.shape[0]), Kxx)

    # there are three outputs
    latent_t = latent[:N]
    latent_dt = latent[N:N*2]
    latent_dt2 = latent[N*2:]

    # compute approximate derivatives
    approx_dt = np.gradient(latent_t, X[:, 0])
    approx_dt2 = np.gradient(latent_dt, X[:, 0])

    # ==== assert ====
    # ignore the end points as the approximation from numpy is bad there
    # we have to a large a rtol due the approximation error of np.gradient
    np.testing.assert_allclose(approx_dt[2:-2], latent_dt[2:-2], atol=0.1)
    #np.testing.assert_allclose(approx_dt2[2:-2], latent_dt2[2:-2], atol=0.1)


@pytest.mark.parametrize('N', [20])
@pytest.mark.parametrize('rbf_ls', [1.0])
@pytest.mark.parametrize('rbf_var', [0.27])
def test_FirstOrderDerivativeKernel_2D(rbf_ls, rbf_var, N, regression_1d_data):
    # ==== Arrange ====
    x1, x2 = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    X = np.hstack([np.reshape(x1, [N*N, 1]), np.reshape(x2, [N*N, 1])])
    N_grid = N
    N = N * N

    # ==== Act ====
    # we construct an derivative kernel, take a sample and then assert 
    #   that the sampled first and second order matches the explictely
    #   computed gradients

    base_kernel = ScaleKernel(RBF(input_dim = 2, lengthscales = [rbf_ls, rbf_ls]), rbf_var)

    kern = FirstOrderDerivativeKernel(
        FirstOrderDerivativeKernel(base_kernel, input_index = 0),
        input_index = 1,
        parent_output_dim = 2
    )

    Kxx = kern.K(X, X)

    # check out samples
    latent = np.random.multivariate_normal(np.zeros(Kxx.shape[0]), Kxx)

    # there are three outputs
    latent_t = latent[:N]
    latent_dt = latent[N:N*2].reshape([N_grid, N_grid])
    latent_ds = latent[N*2:N*3].reshape([N_grid, N_grid])
    latent_dtds = latent[N*3:].reshape([N_grid, N_grid])

    # compute approximate derivatives
    approx_dt, approx_ds = np.gradient(
        latent_t.reshape(N_grid, N_grid), 
        x1[0], 
        x2[:, 0]
    )

    approx_dt2, approx_dsdt = np.gradient( approx_dt, x1[0], x2[:, 0])

    approx_dtds, approx_ds2 = np.gradient( approx_ds, x1[0], x2[:, 0])


    # ==== assert ====
    # ignore the end points as the approximation from numpy is bad there
    # we have to a large a rtol due the approximation error of np.gradient
    np.testing.assert_allclose(
        approx_dt[2:-2, 2:-2], 
        latent_dt[2:-2, 2:-2], 
        atol=0.1
    )

    np.testing.assert_allclose(
        approx_ds[2:-2, 2:-2], 
        latent_ds[2:-2, 2:-2], 
        atol=0.1
    )

    np.testing.assert_allclose(
        approx_dtds[2:-2, 2:-2], 
        latent_dtds[2:-2, 2:-2], 
        atol=0.1
    )

    np.testing.assert_allclose(
        approx_dsdt[2:-2, 2:-2], 
        latent_dtds[2:-2, 2:-2], 
        atol=0.1
    )
