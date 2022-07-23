import pytest
import numpy as np

import stgp
from stgp.likelihood import Gaussian
from stgp.kernels import RBF, ScaleKernel
from stgp.approximate_posteriors import GaussianApproximatePosterior
from stgp.models import GP
from stgp.sparsity import NoSparsity

@pytest.fixture
def regression_1d_data(N):
    np.random.seed(0)
    x = np.linspace(-1, 1, N)
    y = np.sin(x) + 0.1*np.random.randn(N)
    return x[:, None], y[:, None]

@pytest.fixture
def regression_2d_data(N):
    np.random.seed(0)
    x1, x2 = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    X = np.hstack([np.reshape(x1, [N*N, 1]), np.reshape(x2, [N*N, 1])])
    
    N = X.shape[0]

    y = np.sin(X[:, 0]) + np.sin(X[:, 1]) + 0.1*np.random.randn(N)
    return X, y[:, None]

@pytest.fixture
def gaussian_likelihood(var):
    return Gaussian(variance=var)

@pytest.fixture
def rbf_1d_kernel(rbf_var, rbf_ls):
    return ScaleKernel(RBF(input_dim=1, lengthscales=np.array([rbf_ls])), variance=rbf_var)

@pytest.fixture
def rbf_2d_kernel(rbf_var, rbf_ls):
    return ScaleKernel(RBF(input_dim=2, lengthscales=np.array([rbf_ls, rbf_ls])), variance=rbf_var)

@pytest.fixture
def gaussian_approximate_posterior(N):
    np.random.seed(0)

    m = np.random.randn(N)[:, None]
    S = np.random.randn(N, N)
    S[np.tril_indices(N)] = 0
    S = S.T @ S + 1e-4*np.eye(N)

    return GaussianApproximatePosterior(dim=N, m = m, S=S)

@pytest.fixture
def gp_prior_1d(N, regression_1d_data, rbf_1d_kernel):
    X, Y = regression_1d_data

    Z = NoSparsity(Z = X)

    latent_gp = GP(
        sparsity = Z, 
        kernel = rbf_1d_kernel
    )

    return latent_gp

@pytest.fixture
def gp_prior_2d(N, regression_2d_data, rbf_2d_kernel):
    X, Y = regression_2d_data

    Z = NoSparsity(Z = X)

    latent_gp = GP(
        sparsity = Z, 
        kernel = rbf_2d_kernel
    )

    return latent_gp


