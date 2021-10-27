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
def test_single_vgp_elbo_gradient_wrt_lengthscale(N, regression_1d_data):
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
    ls_raw = 0.4

    # Test
    prior = GP(
        X,
        kernel=gpax.kernel.RBF(variance=1.0, lengthscales=np.array([ls_raw])),
    )
    model = GP(
        X, 
        Y, 
        prior=prior,
        likelihood=[gpax.likelihood.Gaussian(variance=sigma)], 
        inference='Variational', 
        whiten=False
    )


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

    ls = inv_softplus(ls_raw)
    dNML_dl = (nml(ls+1e-7)-nml(ls))/1e-7

    train_vars = model.vars()
    grad_fn = objax.GradValues(model.get_objective, train_vars)

    grad, val = grad_fn()

    # Assert
    true_val = np.squeeze(dNML_dl)
    test_val = np.squeeze(np.array(grad[1]))

    np.testing.assert_almost_equal(true_val, test_val, decimal=5)





