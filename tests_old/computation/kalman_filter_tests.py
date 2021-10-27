import unittest

import sys
sys.path.append('../../')

import gpjax
from gpjax.kernels import Matern32
from gpjax.likelihoods import GaussianLikelihood

from gpjax.computation import kalman_filter, kalman_loop_store_intermediate

import numpy as np
import numpy.testing as npt
import scipy

import jax
import jax.numpy as jnp


class TestKalmanFilter(unittest.TestCase):
    def __get_kernel_and_likelihood(self):
        kernel = Matern32(params={
            'lengthscale': np.log(1.0),
            'variance': np.log(1.0)
        }, trainable=False) 

        likelihood = GaussianLikelihood(params={'variance': np.log(1.0)}, trainable=False) 

        return kernel, likelihood


    def test_matern32_chol_diag(self):
        #SETUP
        N = 100
        X = np.linspace(0, 1, N)[:, None]
        Y = np.sin(10*X)+np.random.randn(N)[:, None]

        kernel, likelihood = self.__get_kernel_and_likelihood()

        dt = jnp.concatenate([np.array([0.0]), np.diff(X[:, 0])])
        mask = np.zeros(Y.shape[0], dtype=bool)
        mask = jnp.array(mask)

        #TRUE
        Q = kernel.K(X, X)+np.eye(N)*likelihood.variance
        Q_chol = np.linalg.cholesky(Q)

        chol_diag_true = np.diag(Q_chol)


        #RESULT
        _, _, _, alpha, beta, chol_diag = kalman_loop_store_intermediate(jnp.array(Y), N, dt, kernel, likelihood, mask)
        

        npt.assert_almost_equal(chol_diag, chol_diag_true, decimal=7)

    def test_matern32_beta(self):
        #SETUP
        N = 100
        X = np.linspace(0, 1, N)[:, None]
        Y = np.sin(10*X)+np.random.randn(N)[:, None]

        kernel, likelihood = self.__get_kernel_and_likelihood()

        dt = jnp.concatenate([np.array([0.0]), np.diff(X[:, 0])])
        mask = np.zeros(Y.shape[0], dtype=bool)
        mask = jnp.array(mask)

        #TRUE
        Q = kernel.K(X, X)+np.eye(N)*likelihood.variance
        Q_chol = np.linalg.cholesky(Q)

        beta_true = np.linalg.solve(Q_chol, Y)


        #RESULT
        _, _, _, alpha, beta, chol_diag = kalman_loop_store_intermediate(jnp.array(Y), N, dt, kernel, likelihood, mask)

        npt.assert_almost_equal(np.squeeze(beta), np.squeeze(beta_true), decimal=7)

if __name__ == '__main__':
    with jax.disable_jit():
        unittest.main()
