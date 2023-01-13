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
from stgp.computation.parameter_transforms import correlation_transform, get_correlation_cholesky


@pytest.fixture
def regression_1d_diff_obs_data(N):
    np.random.seed(0)

    f = lambda x: np.sin(10*x)
    df = lambda x: np.cos(10*x)*10

    x = np.linspace(0, 1, N)
    y = f(x) + 0.01*np.random.rand(N)
    dy = df(x) + 0.01*np.random.rand(N)
    Y_all = np.hstack([y[:, None], dy[:, None]])
    return x[:, None], Y_all

@pytest.fixture
def regression_2d_diff_obs_data(N):
    """ Create grid of size NxN. """
    np.random.seed(0)

    x1, x2 = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    X = np.hstack([np.reshape(x1, [N*N, 1]), np.reshape(x2, [N*N, 1])])

    N = X.shape[0]

    # Construct data
    f = lambda x1, x2: np.sin(10*x1*x2)
    df_x1 = lambda x1, x2: np.cos(10*x1 * x2)* 10 * x2
    df_x2 = lambda x1, x2: np.cos(10*x1 * x2)* 10 * x1

    np.random.seed(0)
    y = f(X[:, 0], X[:, 1]) + np.random.randn(N)* 0.01
    dy_x1 = df_x1(X[:, 0], X[:, 1]) + np.random.randn(N)* 0.01
    dy_x2 = df_x2(X[:, 0], X[:, 1]) + np.random.randn(N)* 0.01

    Y = np.hstack([y[:, None], dy_x1[:, None]*np.NaN, dy_x2[:, None]*np.NaN, dy_x1[:, None] * np.NaN])

    return X, Y


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
def multi_output_timeseries(P, N, seed=0):
    np.random.seed(seed)

    x = np.linspace(0, 1, N)
    X = x[:, None]

    Q = int(P*(P-1)/2)
    z = correlation_transform(np.random.randn(Q)*1.0, 1.0)

    # Random correlation cholesky
    L = np.array(get_correlation_cholesky(z, P, Q))
    R = L @ L.T

    # Random (postive) variances
    V = np.exp(np.random.uniform(0, 1, P))

    # Get random covariance cholesky
    W = np.diag(V) @ L

    # Random lengthscales for latent functions
    lengthscales = 0.05*np.random.random(P)

    # Random output noise for each output
    lik_noise = np.random.random(P)*0.1

    #Get P samples from GPs with unit variance
    latent_fn = []
    for i in range(P):
        K= stgp.kernels.RBF(
            lengthscales=np.array([lengthscales[i]]), 
        )
        K_xx = K.K(X, X) + np.eye(N)*1e-7
        latent = np.random.multivariate_normal(np.zeros(N), K_xx)
        latent_fn.append(latent[:, None])

    latents = np.concatenate(latent_fn, axis=1)

    #Create correlatation between the P samples
    correlated_latents = (W @ latents.T).T

    #Add output specific noise
    for i in range(P):
        correlated_latents[:, i] = correlated_latents[:, i] + lik_noise[i]*np.random.randn(N)

    Y = correlated_latents

    return X, Y

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
def gp_model_1d(N, gaussian_likelihood, regression_1d_data, rbf_1d_kernel):
    X, Y = regression_1d_data
    data = Data(X, Y)

    Z = NoSparsity(Z = X)

    latent_gp = GP(
        sparsity = Z, 
        kernel = rbf_1d_kernel
    )
        
    # Create Model
    m = stgp.models.GP(
        data = data, 
        kernel=rbf_1d_kernel,
        likelihood = gaussian_likelihood
    )

    return m


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


