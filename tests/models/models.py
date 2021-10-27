import pytest

import gpax
from gpax.factory.models import get_gpr
from gpax.model import GP
from gpax.model import GP
import objax

import numpy as np
import scipy as sp

from ..common_fixtures import regression_1d_data

import jax
from jax.config import config


config.update('jax_disable_jit', True)
config.update("jax_enable_x64", True)


@pytest.mark.parametrize('N', [30])
def test_single_gpr_ml_gradient_wrt_lengthscale(N, regression_1d_data):
    """ Compares gradients from Jax using finite differences. """
    softplus = lambda x: np.log(1+np.exp(x))
    inv_softplus = lambda x: np.log(np.exp(x) - 1.)
    logistic = lambda x: np.exp(x) / (1+np.exp(x))

    # Setup
    X, Y = regression_1d_data

    # lik hyper params
    sigma = 1.0

    # Kernel hyperparams
    tau = (X - X.T)**2
    var = 1.0

    # Test
    model = get_gpr(X, Y)

    def nml(ls):
        pos_ls = softplus(ls)
        K_f = var*np.exp(-0.5 * tau / pos_ls**2)

        jitter = 1e-5

        K_y = K_f + sigma*np.eye(N)
        K_y_chol = np.linalg.cholesky(K_y+jitter)

        A = sp.linalg.cho_solve((K_y_chol, True), Y) 

        ML = -0.5 * Y.T @ A - 0.5 * np.sum(np.log(np.square(np.diag(K_y_chol)))) - (N/2)*np.log(2*np.pi)
        NML = -ML

        return NML

    ls = inv_softplus(1.0)
    dNML_dl = (nml(ls+1e-7)-nml(ls))/1e-7

    train_vars = model.vars()
    grad_fn = objax.GradValues(model.get_objective, train_vars)

    grad, val = grad_fn()

    # Assert
    true_val = np.squeeze(dNML_dl)
    test_val = np.squeeze(np.array(grad[1]))

    np.testing.assert_almost_equal(true_val, test_val, decimal=5)

@pytest.mark.parametrize('N', [30])
def test_independent_gpr_ml_gradient_wrt_lengthscale(N, regression_1d_data):
    # Setup
    X, Y = regression_1d_data

    # Setup Single GP

    m = GP(X, Y,  likelihood=[gpax.likelihood.Gaussian(variance=0.1)], kernel=[gpax.kernel.RBF(lengthscales=[0.1], variance=0.2)], inference='Variational')

    train_vars = m.vars()
    grad_fn = objax.GradValues(m.get_objective, train_vars)

    single_grad, single_val = grad_fn()

    # Setup Multiple Independent GP

    P = 3

    Y = np.concatenate([Y for i in range(P)], axis=1)

    m = GP(X, Y,  likelihood=[gpax.likelihood.Gaussian(variance=0.1) for p in range(P)], kernel=[gpax.kernel.RBF(lengthscales=[0.1], variance=0.2) for p in range(P)], inference='Variational')

    train_vars = m.vars()
    grad_fn = objax.GradValues(m.get_objective, train_vars)

    joint_grad, joint_val = grad_fn()


    # Assert

    np.testing.assert_almost_equal(
        np.array(single_val)*P, 
        np.array(joint_val), 
        decimal=5
    )

    for i in range(len(single_grad)):
        for p in range(P):
            np.testing.assert_almost_equal(
                np.array(single_grad[i]), 
                np.array(joint_grad[i*len(single_grad)+p]), 
                decimal=5
            )







