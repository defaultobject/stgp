""" 
Unittests for testing compatability against gpflow 2
"""
import jax 
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import jax.numpy as jnp
import objax

import gpflow

import pytest
import numpy as np
import scipy

import stgp
from stgp import settings
from stgp.computation.elbos.kullback_leiblers import gaussian_cholesky_kl, gaussian_kl, whitened_gaussian_kl
from stgp.dispatch import evoke
from stgp.sparsity import NoSparsity
from stgp.transforms.pdes import DifferentialOperatorJoint
from stgp.approximate_posteriors import FullGaussianApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior
from stgp.kernels.diff_op import SecondOrderDerivativeKernel_2D
from stgp.kernels import RBF, ScaleKernel
from stgp.likelihood import Gaussian
from stgp.trainers import GradDescentTrainer, ScipyTrainer, NatGradTrainer

import gpflow
from gpflow.models import VGP, GPR, SGPR, SVGP
from gpflow.optimizers import NaturalGradient
import tensorflow as tf
from gpflow.optimizers.natgrad import XiSqrtMeanVar

from ..common_fixtures import regression_1d_data, rbf_1d_kernel, gaussian_likelihood, full_posterior_joint_model_no_sparsity, gp_prior_1d, gaussian_likelihood, gp_model_1d

def get_random_correlation_matrix(N, jit=1e-8):
    num_random_elements = int(N*(N+1)/2)
    L = np.zeros([N, N])
    a = np.random.randn(num_random_elements)
    L[np.tril_indices(N, 0)] = a
    A = L @ L.T

    #add jitter
    A = A + jit*np.eye(N)
    L = np.linalg.cholesky(A)
    a = L[np.tril_indices(N, 0)]
    return a,  L, L @ L.T

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [50])
@pytest.mark.parametrize('NS', [200])
@pytest.mark.parametrize('lik_var', [0.2])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
def test__vgp__with_whitening(seed, N, NS, regression_1d_data, lik_var, rbf_ls, rbf_var):
    # ==== Arrange ====
    np.random.seed(0)

    settings.jitter = 1e-6

    X, Y = regression_1d_data
    XS = np.linspace(0, 1, NS)[:, None]

    # random approx posterior values
    m_rand = np.random.randn(N)[:, None]
    _, S_sqrt, S = get_random_correlation_matrix(N)
    q_sqrt = S_sqrt[np.tril_indices(N, 0)]

    # construct stgp model
    g_q = GaussianApproximatePosterior(dim=N, m=jnp.array(m_rand), S_chol_vec=jnp.array(q_sqrt))
    mf_q = MeanFieldApproximatePosterior(approximate_posteriors=[g_q])

    m_stgp = stgp.models.GP(
        data = stgp.data.Data(X, Y), 
        kernel=ScaleKernel(RBF(lengthscales=[rbf_ls]), rbf_var),
        likelihood = [Gaussian(variance=lik_var)],
        inference='Variational',
        approximate_posterior = mf_q,
        whiten=True
    )

    # construct gpflow mode
    inducing_variable = X

    m_gpflow = SVGP(
        kernel = gpflow.kernels.RBF(lengthscales=rbf_ls, variance=rbf_var),
        likelihood = gpflow.likelihoods.Gaussian(variance=lik_var),
        inducing_variable = inducing_variable,
        whiten = True ,
        q_mu=m_rand, 
        q_sqrt=np.array([S_sqrt])
    )

    # ==== Act ====

    # collect elbos
    gpflow_elbo = -m_gpflow.elbo((X, Y)).numpy()
    stgp_elbo = m_stgp.get_objective()

    # collect predictions
    gpflow_pred_mu, gpflow_pred_var = m_gpflow.predict_f(XS)
    stgp_pred_mu, stgp_pred_var = m_stgp.predict_f(XS)

    # === Assert  ===

    np.testing.assert_allclose(gpflow_elbo, stgp_elbo, rtol=1e-04)
    np.testing.assert_allclose(np.squeeze(gpflow_pred_mu), np.squeeze(stgp_pred_mu), rtol=1e-04)
    np.testing.assert_allclose(np.squeeze(gpflow_pred_var), np.squeeze(stgp_pred_var), rtol=1e-04)

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [100])
@pytest.mark.parametrize('NS', [200])
@pytest.mark.parametrize('lik_var', [0.2])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
def test__vgp__no_whitening(seed, N, NS, regression_1d_data, lik_var, rbf_ls, rbf_var):
    # ==== Arrange ====
    np.random.seed(0)

    settings.jitter = 1e-7

    X, Y = regression_1d_data
    XS = np.linspace(0, 1, NS)[:, None]

    # random approx posterior values
    m_rand = np.random.randn(N)[:, None]
    _, S_sqrt, S = get_random_correlation_matrix(N)
    q_sqrt = S_sqrt[np.tril_indices(N, 0)]

    # construct stgp model
    g_q = GaussianApproximatePosterior(dim=N, m=jnp.array(m_rand), S_chol_vec=jnp.array(q_sqrt))
    mf_q = MeanFieldApproximatePosterior(approximate_posteriors=[g_q])

    m_stgp = stgp.models.GP(
        data = stgp.data.Data(X, Y), 
        kernel=ScaleKernel(RBF(lengthscales=[rbf_ls]), rbf_var),
        likelihood = [Gaussian(variance=lik_var)],
        inference='Variational',
        approximate_posterior = mf_q
    )

    # construct gpflow mode
    inducing_variable = X

    m_gpflow = SVGP(
        kernel = gpflow.kernels.RBF(lengthscales=rbf_ls, variance=rbf_var),
        likelihood = gpflow.likelihoods.Gaussian(variance=lik_var),
        inducing_variable = inducing_variable,
        whiten = False ,
        q_mu=m_rand, 
        q_sqrt=np.array([S_sqrt])
    )

    # ==== Act ====

    # collect elbos
    gpflow_elbo = -m_gpflow.elbo((X, Y)).numpy()
    stgp_elbo = m_stgp.get_objective()

    # collect predictions
    gpflow_pred_mu, gpflow_pred_var = m_gpflow.predict_f(XS)
    stgp_pred_mu, stgp_pred_var = m_stgp.predict_f(XS)

    # === Assert  ===

    np.testing.assert_allclose(gpflow_elbo, stgp_elbo, rtol=1e-04)
    np.testing.assert_allclose(np.squeeze(gpflow_pred_mu), np.squeeze(stgp_pred_mu), rtol=1e-04)
    np.testing.assert_allclose(np.squeeze(gpflow_pred_var), np.squeeze(stgp_pred_var), rtol=1e-04)

@pytest.mark.slow
@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [100])
@pytest.mark.parametrize('NS', [200])
@pytest.mark.parametrize('lik_var', [0.2])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
@pytest.mark.parametrize('epochs', [1000])
@pytest.mark.parametrize('lr', [0.01])
def test__vgp__no_whittening_after_adam(seed, epochs, lr, N, NS, regression_1d_data, lik_var, rbf_ls, rbf_var):
    # ==== Arrange ====
    np.random.seed(0)

    settings.jitter = 1e-6

    X, Y = regression_1d_data
    XS = np.linspace(0, 1, NS)[:, None]

    # random approx posterior values
    m_rand = np.random.randn(N)[:, None]
    _, S_sqrt, S = get_random_correlation_matrix(N)
    q_sqrt = S_sqrt[np.tril_indices(N, 0)]

    # construct stgp model
    g_q = GaussianApproximatePosterior(dim=N, m=jnp.array(m_rand), S_chol_vec=jnp.array(q_sqrt))
    mf_q = MeanFieldApproximatePosterior(approximate_posteriors=[g_q])

    m_stgp = stgp.models.GP(
        data = stgp.data.Data(X, Y), 
        kernel=ScaleKernel(RBF(lengthscales=[rbf_ls]), rbf_var),
        likelihood = [Gaussian(variance=lik_var)],
        inference='Variational',
        approximate_posterior = mf_q
    )

    # construct gpflow mode
    inducing_variable = X

    m_gpflow = SVGP(
        kernel = gpflow.kernels.RBF(lengthscales=rbf_ls, variance=rbf_var),
        likelihood = gpflow.likelihoods.Gaussian(variance=lik_var),
        inducing_variable = inducing_variable,
        whiten = False ,
        q_mu=m_rand, 
        q_sqrt=np.array([S_sqrt])
    )

    # ==== Act ====

    # train stgp

    _, _ = GradDescentTrainer(
        m_stgp, 
        objax.optimizer.Adam,
    ).train(
        lr,
        epochs,
        callback = None
    )

    # train gpflow

    data = (X, Y)
    adam_optimizer = tf.optimizers.Adam(lr)
    svgp_natgrad_loss = m_gpflow.training_loss_closure(data)
    for i in range(epochs):
        adam_optimizer.minimize(svgp_natgrad_loss, m_gpflow.trainable_variables)

    # collect elbos
    gpflow_elbo = -m_gpflow.elbo(data).numpy()
    stgp_elbo = m_stgp.get_objective()

    # collect predictions
    gpflow_pred_mu, gpflow_pred_var = m_gpflow.predict_f(XS)
    stgp_pred_mu, stgp_pred_var = m_stgp.predict_f(XS)

    # === Assert  ===

    np.testing.assert_allclose(gpflow_elbo, stgp_elbo, rtol=1e-04)
    np.testing.assert_allclose(np.squeeze(gpflow_pred_mu), np.squeeze(stgp_pred_mu), rtol=1e-04)
    np.testing.assert_allclose(np.squeeze(gpflow_pred_var), np.squeeze(stgp_pred_var), rtol=1e-04)
