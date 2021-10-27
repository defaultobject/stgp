import unittest

import sys
sys.path.append('../../')

import jax 
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_disable_jit', False)
import jax.numpy as jnp

import gpjax
from gpjax.computation import positive_transform, lower_triangle, cholesky_solve,log_chol_matrix_det, correlation_transform, get_correlation_cholesky, kf_cholesky_solve_trace
from gpjax.kernels import Matern32
from gpjax.likelihoods import GaussianLikelihood

import numpy as np
import numpy.testing as npt
import scipy
from scipy.stats import random_correlation

import os
import subprocess

def get_random_correlation_matrix(N):
    num_random_elements = int(N*(N+1)/2)
    L = np.zeros([N, N])
    a = np.random.randn(num_random_elements)
    L[np.tril_indices(N, 0)] = a

    return  L @ L.T

class TestGeneral(unittest.TestCase):
    def test_positive_transform__makes_positive(self):
        a = -2.0

        result = positive_transform(a)

        self.assertTrue(
            result >= 0
        )

    def test_positive_transform__stays_positive(self):
        a = 2.0

        result = positive_transform(a)

        self.assertTrue(
            result >= 0
        )

    def test_lower_triangle__is_lower_triangle__N_2(self):
        N = 2

        vec = [1.0, 2.0, 3.0]
        true = np.array([[vec[0], 0], [vec[1], vec[2]]])

        result = lower_triangle(vec, N)

        self.assertTrue(
            np.array_equal(true, result)
        )

    def test_cholesky_solve__diagional_same_as_numpy(self):
        np.random.seed(0)

        N = 5
        A = np.eye(N)
        A_chol = np.linalg.cholesky(A)
        b = np.random.randn(N)[:, None]

        true = np.linalg.solve(A, b)
        result = cholesky_solve(A_chol, b)

        np.testing.assert_array_almost_equal(true, result)

    def test_cholesky_solve__diagional_and_matrix_same_as_numpy(self):
        np.random.seed(0)

        N = 5
        A = np.eye(N)
        A_chol = np.linalg.cholesky(A)
        b = np.random.randn(N)[:, None]
        b = b @ b.T

        true = np.linalg.solve(A, b)
        result = cholesky_solve(A_chol, b)

        np.testing.assert_array_almost_equal(true, result)

    def test_cholesky_solve__full_matrix_same_as_numpy(self):
        np.random.seed(0)
        N = 100
        A  = get_random_correlation_matrix(N)
        A_chol = np.linalg.cholesky(A+1e-8*np.eye(N))

        b = np.random.randn(N)[:, None]

        #true = np.linalg.solve(A+1e-8*np.eye(N), b)

        true = scipy.linalg.solve_triangular(A_chol.T, scipy.linalg.solve_triangular(A_chol, b, lower=True), lower=False)
        result = cholesky_solve(A_chol, b)

        np.testing.assert_array_almost_equal(true, result, decimal=4)


    def test_log_chol_matrix_det__against_numpy__on_diag(self):
        N = 10
        A = np.eye(N)

        true = np.linalg.slogdet(A)[1]

        result = log_chol_matrix_det(A)
        
        np.testing.assert_almost_equal(true, result)

    def test_log_chol_matrix_det__against_numpy__on_random(self):
        np.random.seed(0)
        A = random_correlation.rvs((.5, .8, 1.2, 1.5))

        true = np.linalg.slogdet(A)[1]

        result = log_chol_matrix_det(np.linalg.cholesky(A))

        np.testing.assert_almost_equal(true, result)

    def test_log_chol_matrix_det__against_numpy__on_large_random(self):
        np.random.seed(0)
        N = 100
        A  = get_random_correlation_matrix(N)+1e-8*np.eye(N)


        true = np.linalg.slogdet(A)[1]

        result = np.array(log_chol_matrix_det(np.linalg.cholesky(A)))

        np.testing.assert_almost_equal(true, result, decimal=5)

    def test_get_correlation_cholesky__same_as_julia_small(self):
        #x = np.array([6.297832627922249, 8.521470297799327, 3.021130558455025])
        P = 2 #number of outputs dimensions
        Q = int(P*(P-1)/2) #Q is the number of latent functions

        x = 10*np.random.randn(Q)
        
        z = correlation_transform(x, 1.0)
        R_chol = get_correlation_cholesky(z, P, Q)
        R = R_chol @ R_chol.T

        julia = '/Applications/Julia-1.4.app/Contents/Resources/julia/bin/julia '
        file_name='corrmap.jl'
        args = '"'+np.array2string(x, separator=', ')+'"'

        pipe = subprocess.Popen(julia + ' ' + file_name + ' ' + args, shell=True, stdout=subprocess.PIPE)
        R_true = pipe.communicate()[0].decode()

        R_true = R_true[1:]
        R_true = R_true[:-1]
        R_true = np.array([a.split(' ') for a in R_true.split('; ')]).astype(np.float)

        np.testing.assert_array_almost_equal(R_true, R)
        
    def test_get_correlation_cholesky__same_as_julia_big(self):
        #x = np.array([6.297832627922249, 8.521470297799327, 3.021130558455025])
        P = 30 #number of outputs dimensions
        Q = int(P*(P-1)/2) #Q is the number of latent functions

        x = 10*np.random.randn(Q)
        
        z = correlation_transform(x, 1.0)
        R_chol = get_correlation_cholesky(z, P, Q)
        R = R_chol @ R_chol.T

        julia = '/Applications/Julia-1.4.app/Contents/Resources/julia/bin/julia '
        file_name='corrmap.jl'
        args = '"'+np.array2string(x, separator=', ')+'"'

        pipe = subprocess.Popen(julia + ' ' + file_name + ' ' + args, shell=True, stdout=subprocess.PIPE)
        R_true = pipe.communicate()[0].decode()

        R_true = R_true[1:]
        R_true = R_true[:-1]
        R_true = np.array([a.split(' ') for a in R_true.split('; ')]).astype(np.float)

        np.testing.assert_array_almost_equal(R_true, R)

    def __get_kernel_and_likelihood(self):
        kernel = Matern32( lengthscales=np.array([np.log(1.0)]), variances=np.array([np.log(1.0)]), trainable=False) 

        likelihood = GaussianLikelihood(variances=np.log(1.0), trainable=False) 

        return kernel, likelihood

    def __test_matern32_solve_cholesky_trace(self):
        #SETUP
        N = 100
        X = np.linspace(0, 1, N)[:, None]
        Y = np.sin(10*X)+np.random.randn(N)[:, None]
        Y = np.concatenate([Y, Y, Y], axis=1)

        kernel, likelihood = self.__get_kernel_and_likelihood()
        likelihood = GaussianLikelihood(variances=np.log(1.0), trainable=False) 

        dt = jnp.concatenate([np.array([0.0]), np.diff(X[:, 0])])
        mask = np.zeros(Y.shape[0], dtype=bool)
        mask = jnp.array(mask)

        #TRUE
        Q = kernel.K(X, X)
        Q_chol = np.linalg.cholesky(Q)
        L = Q+Q
        L_chol = np.linalg.cholesky(L)

        A_true = np.trace(np.linalg.solve(Q, L))


        #RESULT
        A = kf_cholesky_solve_trace(jnp.array(L_chol), N, dt, kernel, likelihood, mask)

        npt.assert_almost_equal(np.squeeze(A), np.squeeze(A_true), decimal=7)

if __name__ == '__main__':
    with jax.disable_jit():
        unittest.main()


