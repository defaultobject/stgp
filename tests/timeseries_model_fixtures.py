""" Unittests for recovering batch models. """
import jax 
import objax

import pytest
import numpy as np
import scipy

import stgp
from stgp import settings
from stgp.models import GP
from stgp.computation.elbos.kullback_leiblers import gaussian_cholesky_kl, gaussian_kl, whitened_gaussian_kl
from stgp.dispatch import evoke
from stgp.sparsity import NoSparsity
from stgp.approximate_posteriors import FullGaussianApproximatePosterior, MeanFieldApproximatePosterior
from stgp.kernels import RBF, ScaleKernel
from stgp.likelihood import Gaussian
from stgp.kernels.diff_op import SecondOrderDerivativeKernel_2D
from stgp.trainers import NatGradTrainer

from .common_fixtures import regression_1d_data, rbf_1d_kernel, gaussian_likelihood, full_posterior_joint_model_no_sparsity, gp_prior_1d, gaussian_likelihood, gp_model_1d, multi_output_timeseries

@pytest.fixture
def vgp(regression_1d_data, lik_var, rbf_ls, rbf_var, whiten):
    X, Y = regression_1d_data
    data = stgp.data.Data(X, Y)


    # Create Model
    m = stgp.models.GP(
        data = data, 
        kernel=ScaleKernel(RBF(lengthscales=[rbf_ls]), rbf_var),
        likelihood = [Gaussian(variance=lik_var)],
        inference='Variational',
        whiten=whiten
    )

    return m

@pytest.fixture
def gp(regression_1d_data, lik_var, rbf_ls, rbf_var):
    X, Y = regression_1d_data
    data = stgp.data.Data(X, Y)

    # Create Model
    m = stgp.models.GP(
        data = data, 
        kernel=ScaleKernel(RBF(lengthscales=[rbf_ls]), rbf_var),
        likelihood = [Gaussian(variance=lik_var)]
    )

    return m

@pytest.fixture
def lmc(P, multi_output_timeseries, lik_var, rbf_ls, rbf_var):
    # assuming full rank
    Q = P

    X, Y = multi_output_timeseries
    data = stgp.data.Data(X, Y)

    
    # Construct Latent GPs
    Z = [stgp.sparsity.NoSparsity(X) for q in range(Q)]

    latent_kernels = [ScaleKernel(RBF(lengthscales=[rbf_ls]), rbf_var) for q in range(Q)]
    latent_gps = [
        stgp.models.GP(sparsity=Z[q], kernel=latent_kernels[q]) for q in range(Q)
    ] 

    np.random.seed(0)
    W = np.random.randn(P, Q)
    prior = stgp.transforms.multi_output.LMC(latent_gps, W = W, output_dim = P)

    # Create Model
    m = stgp.models.GP(
        data = data, 
        prior = prior
    )

    return m

@pytest.fixture
def vi_fp_lmc(P, multi_output_timeseries, lik_var, rbf_ls, rbf_var, whiten):
    # assuming full rank
    Q = P

    X, Y = multi_output_timeseries
    data = stgp.data.Data(X, Y)

    
    # Construct Latent GPs
    Z = [stgp.sparsity.NoSparsity(X) for q in range(Q)]

    latent_kernels = [ScaleKernel(RBF(lengthscales=[rbf_ls]), rbf_var) for q in range(Q)]
    latent_gps = [
        stgp.models.GP(sparsity=Z[q], kernel=latent_kernels[q]) for q in range(Q)
    ] 

    np.random.seed(0)
    W = np.random.randn(P, Q)
    prior = stgp.transforms.multi_output.LMC(latent_gps, W = W, output_dim = P)

    # Create Model
    m = stgp.models.GP(
        data = data, 
        prior = prior,
        inference='Variational',
        approximate_posterior = FullGaussianApproximatePosterior(dim = X.shape[0] * prior.base_prior.output_dim),
        whiten=whiten

    )

    return m
