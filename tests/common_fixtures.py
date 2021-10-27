import pytest
import numpy as np

import gpax
from gpax.likelihood import Gaussian
from gpax.kernel import RBF
from gpax.approximate_posteriors import GaussianApproximatePosterior

@pytest.fixture
def regression_1d_data(N):
    np.random.seed(0)
    x = np.linspace(-1, 1, N)
    y = np.sin(x) + 0.1*np.random.randn(N)
    return x[:, None], y[:, None]

@pytest.fixture
def gaussian_likelihood(var):
    return Gaussian(variance=var)

@pytest.fixture
def rbf_1d_kernel(lengthscale):
    return RBF(input_dim=1, lengthscales=np.array([lengthscale]))

@pytest.fixture
def gaussian_approximate_posterior(N):
    np.random.seed(0)

    m = np.random.randn(N)[:, None]
    S = np.random.randn(N, N)
    S[np.tril_indices(N)] = 0
    S = S.T @ S + 1e-4*np.eye(N)

    return GaussianApproximatePosterior(dim=N, m = m, S=S)


