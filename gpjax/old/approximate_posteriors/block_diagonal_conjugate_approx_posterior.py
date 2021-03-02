import sys

from . import ApproximatePosterior
from . import GaussianApproxPosterior

# from ..models import Model
from .. import Likelihood
from .. import Distribution


from ..inference import StateSpace
from ..data import Data, ListData, PlaceholderData, ST_PlaceholderData

# from ..models.st_sde_gp import ST_SDE_GP
from ..decorators import *


from ..initializers import vectorized_lower_triangular_cholesky

from ..likelihoods import (
    LMC_Likelihood,
    BlockDiagonalGaussianLikelihood,
    NaturalBlockDiagonalGaussianLikelihood,
)
from ..distributions import (
    GaussianDistribution,
    DiagonalNaturalGaussianDistribution,
    BlockDiagonalNaturalGaussianDistribution,
)
from ..settings import Settings

from ..computation.conditionals import *
from ..computation.kullback_leiblers import *
from ..computation.expected_log_likelihoods import *

# from ..computation.predictors import *
from ..computation.expectation_approximators import *

from ..data import SpatioTemporalData

import jax
import jax.numpy as np
from jax import jit, partial

import numpy as onp

import typing
from typing import Optional, List, Union, Tuple

import inspect

import warnings


class BlockDiagonalConjugateApproxPosterior(GaussianApproxPosterior):
    def __init__(
        self,
        dim: Optional[List[int]] = None,
        lambda_1: Optional[np.ndarray] = None,
        covar_chol: Optional[np.ndarray] = None,
        distribution: Optional[Distribution] = None,
        key: Optional[jax.random.PRNGKey] = None,
        name: Optional[str] = None,
        data=None,
        sparsity=None,
        conjugate_model=None,
        batch_inference=None,
        kernel=None,
        options={},
    ):
        """
        Args:
            dim: [num_blocks, size_block]
            m: np.ndarray inital mean values for q(.)
            S: np.ndarray inital variance values for q(.)
        """
        # save __init__ arguments as properties of this object
        self.save_inputs_to_properties(locals())

        self.z_shape = sparsity.Z.shape
        self.x_shape = self.data.X[0].shape
        self.dim = [self.x_shape[0], self.z_shape[0]]
        dim = self.dim

        self.name = "BlockDiagonalConjugateApproxPosterior"

        super(BlockDiagonalConjugateApproxPosterior, self).__init__(
            dim=dim, name=self.name
        )

        # self.lambda_1 = self.parameter(val=lambda_1, scope='variational', train=self.trainable, module_name=self.name, param_name='mu')

        if distribution is None:
            lambda_1 = np.expand_dims(1e-5 * onp.ones(self.dim), -1)  # Nx1

            # close to ones
            covar_chol = vectorized_lower_triangular_cholesky(onp.eye(self.dim[1]))
            covar_chol = onp.tile(covar_chol, [self.dim[0], 1])

            self.distribution = NaturalBlockDiagonalGaussianLikelihood(
                lambda_1=lambda_1, covar_chol=covar_chol, meta={"N": self.dim}
            )
        else:
            self.distribution = distribution

        if self.key is None:
            self.key = jax.random.PRNGKey(Settings.seed)

        if inspect.isclass(self.conjugate_model):

            Z_tiled = self.sparsity.get_Z_across_time(
                self.data.get_temporal_locations()[0]
            )
            Nt, Ns, D = Z_tiled.shape[0], Z_tiled.shape[1], Z_tiled.shape[2]

            N = np.prod(self.z_shape)

            Z_tiled_flattened = np.reshape(Z_tiled, (Nt * Ns, D))
            approx_mu_flattened = np.reshape(self.distribution.Y, (Nt * Ns, 1))

            self.conjugate_model = self.conjugate_model(
                X=[Z_tiled_flattened],
                Y=[
                    approx_mu_flattened
                ],  # this will not be jitted properly and is passed as a dummy, Must pass through sep.
                inference=self.batch_inference,
                likelihood=self.distribution,
                options=self.options,
                set_defaults=False,
                sparsity=self.sparsity,
                kernel=self.kernel,
            )

        self.num_prediction_samples = 100
        self.num_samples = 20

        # self.setup()

    @property
    def approx_data(self):
        Z_tiled = self.sparsity.get_Z_across_time(self.data.get_temporal_locations()[0])
        X = [Z_tiled]
        Y = [self.distribution.Y]
        data = ST_PlaceholderData(X, Y)
        data.meta = self.data.meta
        data.X_onp = self.data.X_onp

        return data

    def setup(self):
        pass

    @ensure_data_passed
    def predict_f(
        self,
        data_xs: Data,
        data: Data,
        model: "Model",
        latent: int,
        diagonal_var,
        spatial_predict=True,
        temporal_predict=False,
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Args:
            predict: if true return f^* else f
        """

        # TODO: generalise diagonal _var

        # when predicting with conjugate model with use the approximate data
        approx_data = self.approx_data

        if temporal_predict is True:
            mean, var = self.conjugate_model.predict_y(
                data_xs, approx_data, diagonal_var=False
            )
        else:
            mean, var = self.conjugate_model.posterior(approx_data, diagonal_var=False)

        mean, var = mean[0], var[0]

        return self.spatial_conditional(
            data_xs,
            mean,
            var,
            model,
            latent,
            diagonal_var,
            spatial_predict,
            temporal_predict,
        )

    def spatial_conditional(
        self,
        data_xs: Data,
        mean,
        var,
        model: "Model",
        latent: int,
        diagonal_var,
        spatial_predict=True,
        temporal_predict=False,
    ):

        approx_data = self.approx_data

        # assume that diagonal_var is prediction and non_diagonal_var is posterior
        if spatial_predict:

            # XS_spatial = data_xs.spatial_locations

            prior = model.prior_arr[latent]
            sparsity = model.sparsity
            conditional_fn = Dispatcher.dispatch(
                "conditional", type(prior), None, type(sparsity), True, None
            )

            Ns = approx_data.Y[0].shape[1]

            # do not need to specify time column
            mean = np.reshape(mean, [-1, Ns])
            var = np.reshape(var, [-1, Ns, Ns])

            mean, var = conditional_fn(data_xs, sparsity.Z, mean, var, prior)

        if diagonal_var:
            if not spatial_predict:
                var = np.diagonal(var, 0, 1, 2)
            N = np.prod(mean.shape)
            mean = mean.reshape(N, 1)
            var = var.reshape(N, 1)

        return mean, var

    def get_approx_marginal_likelihood(self, model: "Model"):
        # TODO(ollie): Jit=True does not work here
        return -self.conjugate_model.get_objective(
            data=self.approx_data, return_grad=False, jit=False
        )

    # TODO: numpy raw numpy functions in data ordering
    # @jit_with_scope(static_argnums=[1, 4, 5])
    def predict_y(
        self, data_xs: Data, data: Data, model: "Model", latent: int, diagonal_var
    ) -> Tuple[np.ndarray, np.ndarray]:

        if type(data_xs) == PlaceholderData:
            data_xs = SpatioTemporalData(
                [data_xs.X], [data_xs.Y], [data_xs.X], [data_xs.Y]
            )

        likelihood = model.likelihood

        mean, var = self.predict_f(
            data_xs,
            data,
            model,
            latent,
            diagonal_var,
            spatial_predict=True,
            temporal_predict=True,
        )

        grid_sort_idx = data_xs.order_indexes["grid_sort_idx"]

        if True:
            mean = mean[grid_sort_idx, :]
            var = var[grid_sort_idx, :]

        # return mean, var
        if likelihood.predict_requires_approximation():

            # return likelihood.eval(mean), var
            return predict_mean_field_gauss_hermite_quadrature(
                data_xs, data, model.likelihood, model, self.num_samples
            )

        else:
            return likelihood.predict_mean_var(mean, var)

    def predict_latent(
        self, data_xs: Data, data: Data, model: "Model", latent: int, diagonal_var
    ) -> Tuple[np.ndarray, np.ndarray]:

        if type(data_xs) == PlaceholderData:
            data_xs = SpatioTemporalData(
                [data_xs.X], [data_xs.Y], [data_xs.X], [data_xs.Y]
            )

        likelihood = model.likelihood

        mean, var = self.predict_f(
            data_xs,
            data,
            model,
            latent,
            diagonal_var,
            spatial_predict=True,
            temporal_predict=True,
        )

        grid_sort_idx = data_xs.order_indexes["grid_sort_idx"]

        if True:
            mean = mean[grid_sort_idx, :]
            var = var[grid_sort_idx, :]

        return mean, var
