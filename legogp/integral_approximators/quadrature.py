from .integral_approximator import IntegralApproximator

import jax
import jax.numpy as np
import chex
import objax

import itertools
from numpy.polynomial.hermite import hermgauss


def mvhermgauss(H: int, D: int):
    """
    This function is taken from GPflow: https://github.com/GPflow/GPflow
    Copied here rather than imported so that users don't need to install gpflow and tensorflow to use this library
    LICENSE:
        Copyright The Contributors to the GPflow Project. All Rights Reserved.
        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.
    Return the evaluation locations 'xn', and weights 'wn' for a multivariate
    Gauss-Hermite quadrature.
    The outputs can be used to approximate the following type of integral:
    int exp(-x)*f(x) dx ~ sum_i w[i,:]*f(x[i,:])
    :param H: Number of Gauss-Hermite evaluation points.
    :param D: Number of input dimensions. Needs to be known at call-time.
    :return: eval_locations 'x' (H**DxD), weights 'w' (H**D)
    """
    gh_x, gh_w = hermgauss(H)
    x = np.array(list(itertools.product(*(gh_x,) * D)))  # H**DxD
    w = np.prod(np.array(list(itertools.product(*(gh_w,) * D))), 1)  # H**D
    return x, w

class Quadrature(IntegralApproximator):
    def __init__(self):
        self.num_quadrature_points = 20

    def run(
        self, 
        X,
        Y,
        approximate_posterior, 
        likelihood,
        kernel,
        sparsity
    ):
        #TODO: generalise later
        D = 1

        #x in S**D x D
        #w in S**D 
        x, w = mvhermgauss(self.num_quadrature_points, D)

        const = np.pi**-0.5

        m, S_diag = approximate_posterior.marginal(X, kernel, sparsity)

        # Assert shape is [N x 1]
        chex.assert_rank([Y, m, S_diag], [2, 2, 2])
        chex.assert_equal(Y.shape, m.shape)
        chex.assert_equal(m.shape, S_diag.shape)

        def batched_log_likelihood(y, x_s, mu, sig):
            f = 2.0**0.5 * sig*x_s + mu

            #batch over observations
            ll = jax.vmap(likelihood.log_likelihood, (0, 0), 0)(y, f)

            return np.squeeze(ll)

        res = jax.vmap(batched_log_likelihood, (None, 0, None, None), (0))(Y, x, m, S_diag)

        res =  np.sum(w[:, None] * const * res, axis=0)[:, None]

        chex.assert_equal(Y.shape, res.shape)

        return res

    def predict_run(
        self, 
        XS, 
        X,
        approximate_posterior, 
        likelihood,
        kernel,
        sparsity
    ):

        #TODO: generalise later
        D = 1

        #x in S**D x D
        #w in S**D 
        x, w = mvhermgauss(self.num_quadrature_points, D)

        const = np.pi**-0.5

        m, S_diag = approximate_posterior.predictive_marginal(XS, X, kernel, sparsity)

        # Assert shape is [N x 1]
        chex.assert_rank([XS, m, S_diag], [2, 2, 2])
        chex.assert_equal(XS.shape[0], m.shape[0])
        chex.assert_equal(m.shape, S_diag.shape)

        def _batched_moment(x_s, mu, sig, moment_fn):
            f = 2.0**0.5 * sig*x_s + mu

            #vmap over prediction locations
            ll = jax.vmap(moment_fn, (0), 0)(f)
            return np.squeeze(ll)

        batched_moment = jax.vmap(_batched_moment, (0, None, None, None), (0))

        conditional_var = likelihood.conditional_var(m)
        first_moment = batched_moment(x, m, S_diag, lambda f: likelihood.conditional_mean(f))
        second_moment = batched_moment(x, m, S_diag, lambda f: np.square(likelihood.conditional_mean(f)))

        first_moment =  np.sum(w[:, None] * const * first_moment, axis=0)[:, None]
        second_moment =  np.sum(w[:, None] * const * second_moment, axis=0)[:, None]

        mean = first_moment
        var = conditional_var + second_moment - np.square(first_moment)

        chex.assert_rank(mean, 2)
        chex.assert_equal(XS.shape[0], mean.shape[0])
        chex.assert_equal(mean.shape, var.shape)

        return mean, var




