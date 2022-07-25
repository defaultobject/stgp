import pytest
import numpy as np

import stgp
from stgp.data import Data
from stgp.likelihood import Gaussian
from stgp.kernels import RBF, ScaleKernel
from stgp.approximate_posteriors import GaussianApproximatePosterior
from stgp.models import GP
from stgp.sparsity import NoSparsity
from stgp.transforms import DataLatentPermutation
from stgp.transforms.pdes import DifferentialOperatorJoint, HeatEquation2D
from stgp.kernels.diff_op import SecondOrderDerivativeKernel_2D
from stgp.approximate_posteriors import FullGaussianApproximatePosterior

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
def gaussian_likelihood(lik_var):
    return Gaussian(variance=lik_var)

@pytest.fixture
def rbf_1d_kernel(rbf_var, rbf_ls):
    return ScaleKernel(RBF(input_dim=1, lengthscales=np.array([rbf_ls])), variance=rbf_var)

@pytest.fixture
def rbf_2d_kernel(rbf_var, rbf_ls):
    return ScaleKernel(RBF(input_dim=2, lengthscales=np.array([rbf_ls, rbf_ls])), variance=rbf_var)

@pytest.fixture
def gaussian_dist_mean_var(seed, N):
    np.random.seed(seed)

    m = np.random.randn(N, 1)
    S_chol_vec = np.random.randn(int(N*(N-1)/2))

    S_chol = np.zeros((N, N))
    idx = np.tril_indices(N, 0)
    S_chol[idx] = S_chol_vec

    return m, S_chol

@pytest.fixture
def gaussian_approximate_posterior(N, gaussian_dist_mean_var):
    m, S_chol = gaussian_dist_mean_var

    S = S_chol @ S_chol.T + 1e-4*np.eye(N)

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
    data = Data(X, Y)

    Z = NoSparsity(Z = X)

    latent_gp = GP(
        sparsity = Z, 
        kernel = rbf_2d_kernel
    )

    return data, latent_gp


@pytest.fixture
def full_posterior_joint_model_no_sparsity(N, gp_prior_2d, gaussian_likelihood):

    # When we are 2d N is a square
    N = N*N

    data, gp_prior_2d = gp_prior_2d

    diff_op_prior = DifferentialOperatorJoint(
        gp_prior_2d,
        SecondOrderDerivativeKernel_2D(gp_prior_2d.kernel)
    )

    diff_op_prior = HeatEquation2D(diff_op_prior)
    diff_op_prior = DataLatentPermutation(diff_op_prior)

    dim = N * diff_op_prior.output_dim

    m = np.random.randn(dim, 1)
    S_chol_vec = np.random.randn(int(dim*(dim+1)/2))

    q = FullGaussianApproximatePosterior(
        dim = dim,
        m = m,
        S_chol_vec = S_chol_vec
    )

    sparsity = diff_op_prior.base_prior.get_sparsity()

    return q, gaussian_likelihood, diff_op_prior, sparsity, data

@pytest.fixture
def permutation_vectors(num_outputs, N):
    # for each output create N datapoints with value corresponding to the output
    v_latent_data = np.zeros([num_outputs, N])
    v_latent_data += np.arange(num_outputs)[:, None]
    v_latent_data = np.hstack(v_latent_data)[:, None]

    v_data_latent = np.tile(np.arange(num_outputs), [N, 1])
    v_data_latent = np.hstack(v_data_latent)[:, None]

    return v_data_latent, v_latent_data

@pytest.fixture
def permutation_matrices(num_outputs, N):
    mat_latent_data = np.arange((num_outputs*N)**2).reshape(num_outputs*N, num_outputs*N)

    N1 = mat_latent_data.shape[0]
    N2 = mat_latent_data.shape[1]

    idx = np.array([[n+q*N for q in range(num_outputs)] for n in range(N)]).flatten()

    p_rows = []

    for row in range(N1):
        p_rows.append(mat_latent_data[row, idx])

    p_rows = np.array(p_rows)

    mat_data_latent = []

    for col in range(N2):
        mat_data_latent.append(p_rows[idx, col])

    mat_data_latent = np.array(mat_data_latent).T

    return mat_data_latent, mat_latent_data


