""" Unittests for Derivative kernels.  """

import jax 
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', True)

import pytest

import numpy as np
import scipy

import stgp
from stgp.kernels import RBF, ScaleKernel
from stgp.kernels.diff_op import SecondOrderDerivativeKernel_1D


from ..common_fixtures import regression_1d_data, regression_2d_data

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
