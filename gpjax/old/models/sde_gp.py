from . import Model
from .. import Parameter
from .. import Kernel
from .. import Likelihood
from .. import Inference


from ..data import Data, TimeseriesData
from ..distributions import KernelGaussianDistribution

from ..inference import StateSpace
from ..decorators import *
from ..sparsity import *

from ..data.utils import order_timeseries_prediction, unorder_timeseries_prediction

from ..settings import Settings
from ..computation.general import (
    get_computational_primitives,
    kf_cholesky_solve_trace,
    cholesky_solve,
)
from ..computation.kalman_filter import kalman_loop, kalman_loop_store_intermediate
from ..computation.rts_smoother import rts_smoother

from ..likelihoods import GaussianLikelihood

from .. import Dispatcher

import jax.numpy as np
from jax import jit, partial

import numpy as onp

import typing
from typing import Union, List, Optional


@Dispatcher.register("gp_model", StateSpace, None, 1)
class SDE_GP(Model):
    def setup(self):
        self.data = TimeseriesData(self.X, self.Y, self.X_onp, self.Y_onp)
        # if not self.set_defaults: return

        self.prior_arr = []
        for p in range(self.num_outputs):
            prior = KernelGaussianDistribution(
                kernel=self.kernel_arr[p], meta={"X": self.X}
            )
            self.prior_arr.append(prior)

    @return_gradients
    @set_defaults_from_self
    def get_objective(self, data: Optional[Data] = None):
        p = 0
        prior = self.prior_arr[p]
        likelihood = self.likelihood
        Y = data.Y[p]

        kernel = prior.kernel

        # select correct filter based on likelihood
        kf_fn = Dispatcher.dispatch("kalman_filter", type(likelihood))

        neg_log_marg_lik, _, _, _, _, _ = kf_fn(
            Y,
            data.N,
            data.dt,
            kernel,
            likelihood,
            self.sparsity,
            data.mask,
            store_intermediate=False,
        )

        return neg_log_marg_lik

    def filter_and_smooth(
        self,
        Y_all,
        N_all,
        dt_all,
        kernel,
        likelihood,
        sparsity,
        mask,
        store_intermediate,
    ):
        # filter
        kf_fn = Dispatcher.dispatch("kalman_filter", type(likelihood))
        neg_log_marg_lik, filtered_mean, filtered_cov, alpha, beta, chol_diag = kf_fn(
            Y_all,
            N_all,
            dt_all,
            kernel,
            likelihood,
            sparsity,
            mask,
            store_intermediate=store_intermediate,
        )

        # smooth
        smoother_fn = Dispatcher.dispatch("rts_smoother", type(likelihood))
        mu, sig, alpha = smoother_fn(
            Y_all,
            N_all,
            dt_all,
            filtered_mean,
            filtered_cov,
            kernel,
            likelihood,
            sparsity,
            mask,
            alpha,
        )

        return mu, sig

    @set_defaults_from_self
    def posterior(
        self, data: Optional[Data] = None, diagonal_var: Optional[bool] = True
    ):
        prior = self.prior_arr[0]
        likelihood = self.likelihood
        X = data.X[0]
        Y = data.Y[0]
        kernel = prior.kernel

        dt_all = data.dt
        Y_all = Y
        N_all = data.N
        mask = data.mask

        if diagonal_var:
            mu, sig = self.filter_and_smooth(
                Y_all,
                N_all,
                dt_all,
                kernel,
                likelihood,
                self.sparsity,
                mask,
                store_intermediate=True,
            )
        else:
            raise NotImplementedError()

        if diagonal_var:
            mu = np.reshape(mu, [mu.shape[0], 1])
            sig = np.reshape(sig, [sig.shape[0], 1])

        # TODO: for multiple latent support
        return [mu], [sig]

    @ensure_data_passed
    @set_defaults_from_self
    def predict_y(
        self,
        data_xs: Data,
        data: Optional[Data] = None,
        diagonal_var: Optional[bool] = True,
    ):
        prior = self.prior_arr[0]
        likelihood = self.likelihood
        XS = data_xs.X

        kernel = prior.kernel

        # organise data and prediction points into timeseries
        data_all = order_timeseries_prediction(data_xs, data)

        # sort XS for prediction
        dt_all = data_all.dt
        Y_all = data_all.Y[0]
        N_all = data_all.N
        mask = data_all.mask

        if diagonal_var:
            mu, sig = self.filter_and_smooth(
                Y_all,
                N_all,
                dt_all,
                kernel,
                likelihood,
                self.sparsity,
                mask,
                store_intermediate=True,
            )
        else:
            raise NotImplementedError()

        # extract predict locations from the predictions
        mu, sig = unorder_timeseries_prediction(data_all, mu, sig)

        if diagonal_var:
            mu = np.reshape(mu, [mu.shape[0], 1])
            sig = np.reshape(sig, [sig.shape[0], 1])

        return [mu], [sig]

    # @partial(jit, static_argnums=(2))
    def predict_f(self, XS: np.ndarray, diagonal_var: bool):
        raise NotImplementedError("Predict f is not implemented yet")
