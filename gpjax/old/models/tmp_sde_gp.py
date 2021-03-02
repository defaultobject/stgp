from . import Model
from .. import Parameter
from .. import Kernel
from .. import Likelihood
from .. import Inference

from ..inference import StateSpace
from ..decorators import return_gradients
from ..sparsity import *

from ..settings import Settings
from ..computation.general import (
    get_computational_primitives,
    kf_cholesky_solve_trace,
    cholesky_solve,
)
from ..computation.kalman_filter import kalman_loop, kalman_loop_store_intermediate
from ..computation.rts_smoother import rts_smoother

import jax.numpy as np
from jax import jit, partial

import numpy as onp

import typing
from typing import Union, List, Optional


class SDE_GP(Model):
    def order_observations_by_grid(self, X, Y):
        """
        lexsort uses the final column as the primary sort key
            1 -  roll the columns so the time is the last column
            2 - get idx of new ordering
            3 - roll X so that time is the first axis again

        """
        # X is a list

        time_zero = X[0, 0]
        X_spatial = X[X[:, 0] == time_zero][:, 1:]

        grid_size = X_spatial.shape[0]
        time_points = int(X.shape[0] / grid_size)

        # put time axis as last axis os that this is sorted first
        X = onp.roll(X, -1, axis=1)
        idx = onp.lexsort(X.T)

        X = X[idx]

        if Y is not None:
            Y = Y[idx]

        # reset time axis
        X = onp.roll(X, 1, axis=1)
        # reshape for grid structure
        X = np.reshape(X, [time_points, grid_size, X.shape[1]])

        if Y is not None:
            Y = np.reshape(Y, [time_points, grid_size, 1])
        return X, Y

    def get_spatial_locations(self):
        time_zero = self.X_arr[0][0, 0]
        X_spatial = self.X_arr[0][self.X_arr[0][:, 0] == time_zero][:, 1:]
        # add on missing time column so that kernels can still be computed
        X_spatial = np.hstack([onp.zeros([X_spatial.shape[0], 1]), X_spatial])
        return X_spatial

    def organise_one_dimensional_inputs(
        self, X: onp.ndarray, Y: Optional[onp.ndarray]
    ) -> onp.ndarray:
        sort_idx = np.argsort(X, axis=0)

        X = X[sort_idx[:, 0], :]
        if Y is not None:
            Y = Y[sort_idx[:, 0], :]

        return X, Y, sort_idx

    def check_and_fix_inputs(self) -> None:
        """
        Ensures that the data is in the proper format for state space
        for 1d:
            TODO: order timeseries

        for 2d:
            1 - checks that the spatial points are on a grid
            2 - orders observations wrt to time dimension and spatial dimension so that
                k_spatial is consistent across time

        """

        X = self.X_arr[0]
        Y = self.Y_arr[0]

        print("check_and_fix_inputs")

        if len(X.shape) <= 3 and X.shape[1] > 1:
            # if the number of unique times points times the number of unique spatial points equals the total number of points, the data is on a grid
            if type(self.sparsity) == NoSparsity:
                if True or X.shape[0] == int(
                    onp.unique(X[:, 0]).shape[0] * np.sum(onp.unique(X[:, 1:]).shape[0])
                ):
                    if True:
                        print(
                            "Observations are on spatial grid. Sorting to ensure correct format."
                        )

                    X, Y = self.order_observations_by_grid(X, Y)
                else:
                    raise NotImplementedError("Observations must lie on a spatial grid")
            else:
                print(self.sparsity)

        else:
            print("time series")

        self.X_arr = [X]
        self.Y_arr = [Y]

    def setup(self):
        if not self.set_defaults:
            return

        if self.X_arr[0].shape[1] > 1:
            # use spatio temporal kernel
            self.kernel.set_spatial_locations(
                self.get_spatial_locations(), sparsity=self.sparsity
            )

        self.check_and_fix_inputs()

        # self.X = np.array(self.X)
        # self.Y = np.array(self.Y)

        self.N = self.Y_arr[0].shape[0]
        # only get the difference on the time axis
        # self.dt = np.concatenate([np.array([0.0]), onp.diff(self.X[:, 0])])
        self.dt = np.concatenate(
            [np.array([0.0]), onp.diff(onp.unique(self.X_arr[0][:, 0]))]
        )

        # mask for nans in Y
        y_nans = onp.isnan(self.Y_arr[0]).any(axis=1)
        mask = onp.zeros(self.Y_arr[0].shape[0], dtype=bool)
        mask[onp.argwhere(y_nans)] = True

        self.mask = np.array(mask)

        self.no_mask = onp.zeros(self.Y_arr[0].shape[0], dtype=bool)
        self.no_mask = np.array(self.no_mask)

    @return_gradients
    @jit
    def get_objective(self):
        neg_log_marg_lik, _, _, _, _, _ = kalman_loop(
            self.Y_arr[0], self.N, self.dt, self.kernel, self.likelihood, self.mask
        )

        return neg_log_marg_lik

    def organise_predict_time_series(
        self, X: onp.ndarray, Y: onp.ndarray, XS: onp.ndarray, YS: onp.ndarray
    ):
        X_all = onp.concatenate([X, XS], axis=0)
        Y_all = onp.concatenate([Y, YS], axis=0)

        test_idx = np.arange(X.shape[0], X.shape[0] + XS.shape[0])
        X_unique, unique_idx, unique_reverse_idx = onp.unique(
            X_all, return_index=True, return_inverse=True
        )

        X_all = X_all[unique_idx, :]
        Y_all = Y_all[unique_idx, ...]

        X_all, Y_all, xs_sort_idx = self.organise_one_dimensional_inputs(X_all, Y_all)
        xs_sort_idx = np.squeeze(xs_sort_idx)

        return X_all, Y_all, xs_sort_idx, test_idx, unique_idx, unique_reverse_idx

    def predict_timeseries(self, XS: onp.ndarray, diagional_var: bool):
        # get original numpy arrays
        X_onp = self.X_onp[0]
        Y_onp = self.Y_onp[0]

        YS = onp.zeros([XS.shape[0], Y_onp.shape[1]]) * onp.NaN

        # onp.unique returns the idices of the first occurence.
        # we want to use training data over testing, so place in front
        X_all = onp.concatenate([X_onp, XS], axis=0)
        Y_all = onp.concatenate([Y_onp, YS], axis=0)
        test_idx = np.arange(Y_onp.shape[0], Y_onp.shape[0] + YS.shape[0])
        X_unique, unique_idx, unique_reverse_idx = onp.unique(
            X_all, return_index=True, return_inverse=True
        )

        X_all = X_all[unique_idx, :]
        Y_all = Y_all[unique_idx, :]

        X_all, Y_all, xs_sort_idx = self.organise_one_dimensional_inputs(X_all, Y_all)
        xs_sort_idx = np.squeeze(xs_sort_idx)

        dt_all = np.concatenate([np.array([0.0]), onp.diff(X_all[:, 0])])

        # mask for nans in Y
        mask = onp.zeros(Y_all.shape[0], dtype=bool)
        mask[onp.argwhere(onp.isnan(Y_all)[:, 0])] = True
        mask = np.array(mask)

        N_all = Y_all.shape[0]

        if diagional_var:
            (
                neg_log_marg_lik,
                filtered_mean,
                filtered_cov,
                alpha,
                beta,
                chol_diag,
            ) = kalman_loop_store_intermediate(
                Y_all, N_all, dt_all, self.kernel, self.likelihood, mask
            )

            mu, sig, alpha = rts_smoother(
                Y_all,
                N_all,
                dt_all,
                filtered_mean,
                filtered_cov,
                self.kernel,
                self.likelihood,
                mask,
                alpha,
            )

            # unsort output
            # selecting testing region

            mu = mu[xs_sort_idx, :][unique_reverse_idx, :][test_idx, :]
            sig = sig[xs_sort_idx, :][unique_reverse_idx, :][test_idx, :]

            return mu, sig
        else:
            raise NotImplementedError("Non-diagional prediction not implemented yet")

    def predict_spatial(self, XS: onp.ndarray, diagional_var: bool):
        """
        XS is just an array of time locations in Nx1
        """

        assert XS.shape[1] == 1

        X_grid, Y_grid = self.order_observations_by_grid(self.X_onp[0], self.Y_onp[0])

        XS_shape = [XS.shape[0], X_grid.shape[1], X_grid.shape[2]]
        YS_shape = [XS.shape[0], Y_grid.shape[1], Y_grid.shape[2]]

        YS_grid = onp.zeros(YS_shape) * onp.NaN

        training_time_location = np.unique(self.X_onp[0][:, 0])[:, None]

        (
            X_all,
            Y_all,
            xs_sort_idx,
            test_idx,
            unique_idx,
            unique_reverse_idx,
        ) = self.organise_predict_time_series(
            training_time_location, Y_grid, XS, YS_grid
        )

        # X_all is only the time points. join on the spatial points

        X_spatial = X_grid[0, :, 1:]

        # manually for now
        _X_all = []
        for t in X_all:
            t_col = onp.repeat(t, X_spatial.shape[0])[:, None]
            t_grid = onp.hstack([t_col, X_spatial])
            _X_all.append(t_grid)
        X_all = onp.array(_X_all)

        dt_all = np.concatenate([np.array([0.0]), onp.diff(onp.unique(X_all[:, 0]))])

        # mask for nans in Y
        y_nans = onp.isnan(Y_all).any(axis=1)
        mask = onp.zeros(Y_all.shape[0], dtype=bool)
        mask[onp.argwhere(y_nans)] = True

        mask = np.array(mask)

        N_all = Y_all.shape[0]

        print("N_all: ", N_all)
        print("Y_all: ", Y_all.shape)
        print("dt_all: ", dt_all.shape)
        print("mask: ", mask.shape)

        if diagional_var:
            (
                neg_log_marg_lik,
                filtered_mean,
                filtered_cov,
                alpha,
                beta,
                chol_diag,
            ) = kalman_loop_store_intermediate(
                Y_all, N_all, dt_all, self.kernel, self.likelihood, mask
            )

            print("filtered_mean: ", filtered_mean.shape)
            print("filtered_cov: ", filtered_cov.shape)

            mu, sig, alpha = rts_smoother(
                Y_all,
                N_all,
                dt_all,
                filtered_mean,
                filtered_cov,
                self.kernel,
                self.likelihood,
                mask,
                alpha,
            )

            print("Y: ", Y_grid)
            print("predicted mu", mu)

            XS = X_all[xs_sort_idx, :][unique_reverse_idx, :][test_idx, :]

            mu = mu[xs_sort_idx, :][unique_reverse_idx, :][test_idx, :]
            sig = sig[xs_sort_idx, :][unique_reverse_idx, :][test_idx, :]

            print("predicted mu unordered", mu)

            return mu, sig, XS

    # XS has to be a static argument. needs to be an onp array so we can sort it.
    # @partial(jit, static_argnums=(1, 2))
    def predict_y(self, XS: onp.ndarray, diagional_var: bool):
        if self.X_onp[0].shape[1] == 1:
            return self.predict_timeseries(XS, diagional_var)
        else:
            return self.predict_spatial(XS, diagional_var)

    @partial(jit, static_argnums=(2))
    def predict_f(self, XS: np.ndarray, diagional_var: bool):
        raise NotImplementedError("Predict f is not implemented yet")
