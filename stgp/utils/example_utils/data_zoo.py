import numpy as np

import stgp as lego
from stgp.computation.parameter_transforms import (
    correlation_transform,
    get_correlation_cholesky,
)

# Helper functions


def create_grid(x1, x2, y1, y2, n1=10, n2=10):
    y = np.linspace(y1, y2, n2)
    x = np.linspace(x1, x2, n1)

    grid = []
    for i in x:
        for j in y:
            grid.append([i, j])

    return np.array(grid)


# Zoo
def single_output_timeseries(N, NS, seed=0):
    np.random.seed(seed)

    x = np.linspace(0, 1, N)
    y = np.sin(x * 10) + 0.1 * np.random.randn(N)
    X = x[:, None]
    Y = y[:, None]

    XS = np.linspace(-1, 2, 1000)[:, None]

    return XS, X, Y


def multi_output_timeseries(P, N, NS, seed=0):
    np.random.seed(seed)

    x = np.linspace(0, 1, N)
    X = x[:, None]

    Q = int(P * (P - 1) / 2)
    z = correlation_transform(np.random.randn(Q) * 1.0, 1.0)

    # Random correlation cholesky
    L = np.array(get_correlation_cholesky(z, P, Q))
    R = L @ L.T

    # Random (postive) variances
    V = np.exp(np.random.uniform(0, 1, P))

    # Get random covariance cholesky
    W = np.diag(V) @ L

    # Random lengthscales for latent functions
    lengthscales = 0.05 * np.random.random(P)

    # Random output noise for each output
    lik_noise = np.random.random(P) * 0.1

    # Get P samples from GPs with unit variance
    latent_fn = []
    for i in range(P):
        K = lego.kernels.RBF(
            lengthscales=np.array([lengthscales[i]]),
        )
        K_xx = K.K(X, X) + np.eye(N) * 1e-7
        latent = np.random.multivariate_normal(np.zeros(N), K_xx)
        latent_fn.append(latent[:, None])

    latents = np.concatenate(latent_fn, axis=1)

    # Create correlatation between the P samples
    correlated_latents = (W @ latents.T).T

    # Add output specific noise
    for i in range(P):
        correlated_latents[:, i] = correlated_latents[:, i] + lik_noise[
            i
        ] * np.random.randn(N)

    Y = correlated_latents

    XS = np.linspace(-1, 2, 1000)[:, None]

    return XS, X, Y


def single_output_spatial_data(N_time, N_space, NS_time, NS_space, seed=0):
    np.random.seed(seed)

    X = create_grid(-1, 1, -1, 1, N_time, N_space)
    N = X.shape[0]

    y = np.sin(10 * X[:, 0]) + np.sin(10 * X[:, 1]) + 0.01 * np.random.randn(N)
    Y = y[:, None]

    XS = create_grid(-1, 1, -1, 1, NS_time, NS_space)

    return XS, X, Y


def multi_output_spatial_data(P, N_time, N_space, NS_time, NS_space, seed=0):
    np.random.seed(seed)

    X = create_grid(-1, 1, -1, 1, N_time, N_space)
    N = X.shape[0]

    Q = int(P * (P - 1) / 2)
    z = correlation_transform(np.random.randn(Q) * 1.0, 1.0)

    # Random correlation cholesky
    L = np.array(get_correlation_cholesky(z, P, Q))
    R = L @ L.T

    # Random (postive) variances
    V = np.exp(np.random.uniform(0, 1, P))

    # Get random covariance cholesky
    W = np.diag(V) @ L

    # Random lengthscales for latent functions
    lengthscales = [0.5 * np.random.random(2) for p in range(P)]

    # Random output noise for each output
    lik_noise = np.random.random(P) * 0.1

    # Get P samples from GPs with unit variance
    latent_fn = []
    for i in range(P):
        K = lego.kernels.RBF(
            input_dim=2,
            lengthscales=np.array(lengthscales[i]),
        )
        K_xx = K.K(X, X) + np.eye(N) * 1e-7
        latent = np.random.multivariate_normal(np.zeros(N), K_xx)
        latent_fn.append(latent[:, None])

    latents = np.concatenate(latent_fn, axis=1)

    # Create correlatation between the P samples
    correlated_latents = (W @ latents.T).T

    # Add output specific noise
    for i in range(P):
        correlated_latents[:, i] = correlated_latents[:, i] + lik_noise[
            i
        ] * np.random.randn(N)

    XS = create_grid(-1, 1, -1, 1, NS_time, NS_space)

    return XS, X, correlated_latents
