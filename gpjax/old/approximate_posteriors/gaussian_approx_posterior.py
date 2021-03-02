from . import ApproximatePosterior

from .. import Likelihood
from .. import Distribution
from .. import Sparsity
from .. import Dispatcher
from .. import Settings
from .. import Data

from ..likelihoods import GaussianLikelihood
from ..distributions import (
    GaussianDistribution,
    KernelGaussianDistribution,
    WhitenedKernelGaussianDistribution,
    BlockGaussianDistribution,
    BlockWhitenedGaussianDistribution,
)
from ..initializers import vectorized_lower_triangular_cholesky
from ..sparsity import NoSparsity


from ..decorators import *

from ..computation.expectation_approximators import *

import jax
import jax.numpy as np
from jax import jit, partial

import numpy as onp

import typing
from typing import Optional, List, Union, Tuple


class GaussianApproxPosterior(ApproximatePosterior):
    def __init__(
        self,
        dim: int = None,
        m: Optional[np.ndarray] = None,
        S_chol: Optional[np.ndarray] = None,
        distribution: Optional[Distribution] = None,
        key: Optional[jax.random.PRNGKey] = None,
        name: Optional[str] = None,
    ):
        """
        Args:
            m: np.ndarray inital mean values for q(.)
            S: np.ndarray inital variance values for q(.)
        """

        # save __init__ arguments as properties of this object
        self.save_inputs_to_properties(locals())

        self.num_samples = 90
        self.num_prediction_samples = 20

        self.setup()

        super(GaussianApproxPosterior, self).__init__(name=self.name)

    def setup(self) -> None:
        if self.name is None:
            self.name = "GaussianApproxPosterior"

        if self.m is None:
            # close to zero
            self.m = 1e-5 * onp.ones(self.dim)[:, None]  # Nx1

        if self.S_chol is None:
            self.S_chol = vectorized_lower_triangular_cholesky(onp.eye(self.dim))

        if self.distribution is None:
            self.distribution = GaussianDistribution(
                mu=self.m, covariance_chol=self.S_chol, meta={"N": self.dim}
            )

        if self.key is None:
            self.key = jax.random.PRNGKey(Settings.seed)

    def KL(self, X: np.ndarray, distribution: Distribution) -> np.ndarray:
        fun = Dispatcher.dispatch("kullback_leiblers", type(distribution))
        return fun(X, self.distribution, distribution)

    def precomputed_expected_log_likelihood(
        self,
        data: Data,
        likelihood: Likelihood,
        q_mean: np.ndarray,
        q_var: np.ndarray,
        latent=0,
    ) -> np.ndarray:
        """
        useful for when q(f) is already pre-computed
        """
        fun = Dispatcher.dispatch(
            "precomputed_expected_log_likelihoods", type(likelihood), type(self)
        )

        if fun is not False:
            ell = fun(data, likelihood, q_mean, q_var, latent, self)

        else:
            ell = precomputed_mean_field_gauss_hermite_quadrature(
                data, likelihood, [q_mean], [q_var], self.num_samples
            )

        return ell

    def expected_log_likelihood(
        self, data: Data, likelihood: Likelihood, model: "Model", latent=0
    ) -> np.ndarray:

        X = data.X[latent]
        Y = data.Y[latent]

        # likelihood = model.likelihood

        prior = model.prior_arr[latent]
        sparsity = model.sparsity_arr[latent]

        fun = Dispatcher.dispatch(
            "expected_log_likelihoods", type(likelihood), type(self)
        )

        if fun is not False:
            ell = fun(data, likelihood, model, latent, self)

        else:
            ell = gauss_hermite_quadrature(data, likelihood, model, self.num_samples)

        return ell

    @ensure_data_passed
    def predict_f(
        self,
        data_xs: Data,
        data: Data,
        model: "Model",
        latent: int,
        diagonal_var,
        predict=False,
    ) -> Tuple[np.ndarray, np.ndarray]:

        prior = model.prior_arr[latent]
        sparsity = model.sparsity_arr[latent]

        fun = Dispatcher.dispatch(
            "conditional",
            type(prior),
            type(self.distribution),
            type(sparsity),
            diagional_var=diagonal_var,
            predict=predict,
        )

        if type(sparsity) is not NoSparsity:
            X = sparsity.Z
        else:
            X = data.X[0]

        XS = data_xs.X

        return fun(XS, X, prior, self.distribution)

    @ensure_data_passed
    def predict_latent(
        self,
        data_xs: Data,
        data: Data,
        model: "Model",
        latent: int,
        diagonal_var,
        predict=False,
    ) -> Tuple[np.ndarray, np.ndarray]:

        return self.predict_f(data_xs, data, model, latent, diagonal_var, predict)

    # @jit_with_scope(static_argnums=[1, 4, 5])
    def predict_y(
        self, data_xs: Data, data: Data, model: "Model", latent: int, diagonal_var
    ) -> Tuple[np.ndarray, np.ndarray]:
        XS = data_xs.X

        likelihood = model.likelihood

        # return self.predict_f(XS, prior, diagional_var)
        mean, var = self.predict_f(XS, data, model, latent, diagonal_var, predict=True)
        # mu, var = whitened_kernel_gaussian_gaussian_conditional_diagional(XS, prior.X, prior, self.distribution)

        if likelihood.predict_requires_approximation():
            return predict_mean_field_gauss_hermite_quadrature(
                data_xs, data, model.likelihood, model, self.num_samples
            )

        else:
            return likelihood.predict_mean_var(mean, var)
