from . import Data, TimeseriesData

import jax.numpy as np
import numpy as onp

from typing import List


def order_observations_by_grid(X, Y):
    """
    lexsort uses the final column as the primary sort key
        1 -  roll the columns so the time is the last column
        2 - get idx of new ordering
        3 - roll X so that time is the first axis again

    """
    # X is a list

    time_zero = X[0, 0]

    # X_spatial = X[np.isclose(X[:, 0], time_zero)][:,1:]
    X_spatial = X[X[:, 0] == time_zero][:, 1:]

    grid_size = X_spatial.shape[0]
    time_points = int(X.shape[0] / grid_size)

    # put time axis as last axis os that this is sorted first
    X = onp.roll(X, -1, axis=1)
    # sort by time points
    # idx = onp.argsort(X[:, 0])
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

    return idx, X, Y


def reshape_by_grid(X, Y):
    """
    lexsort uses the final column as the primary sort key
        1 -  roll the columns so the time is the last column
        2 - get idx of new ordering
        3 - roll X so that time is the first axis again

    """
    # X is a list

    time_zero = X[0, 0]

    # X_spatial = X[np.isclose(X[:, 0], time_zero)][:,1:]
    X_spatial = X[X[:, 0] == time_zero][:, 1:]

    grid_size = X_spatial.shape[0]
    time_points = int(X.shape[0] / grid_size)

    # reshape for grid structure
    X = np.reshape(X, [time_points, grid_size, X.shape[1]])

    if Y is not None:
        Y = np.reshape(Y, [time_points, grid_size, 1])

    return X, Y


def get_time_points(X):
    points = onp.unique(X[:, 0])
    return onp.sort(points)


class SpatioTemporalData(TimeseriesData):
    @property
    def X(self):
        # TODO generalise
        X = self.X_arr[0]
        return [X]

    @property
    def flattened_X(self):
        X = self.X[0]
        shp = np.shape(X)
        X = np.reshape(X, [shp[0] * shp[1], shp[2]])
        return [X]

    @property
    def flattened_Y(self):
        Y = self.Y[0]
        shp = np.shape(Y)
        Y = np.reshape(Y, [shp[0] * shp[1], shp[2]])
        return [Y]

    @property
    def flattened_mask(self):
        mask = self.mask
        shp = np.shape(mask)
        mask = np.reshape(mask, [shp[0] * shp[1]])
        return [mask]

    @property
    def Y(self):
        Y = self.Y_arr[0]
        return [Y]

    @property
    def spatial_locations(self):
        return self.get_spatial_locations()

    @property
    def mask(self):
        if self.Y_onp is None:
            mask = np.array(np.full((self.X[0].shape[0]), False, dtype=bool))
        else:
            # mask = np.array(self.Y_mask[0][self.order_indexes['argsort']])
            mask = np.array(self.Y_mask[0])
        return mask

    def order(self):
        self.meta["time_points"] = get_time_points(self.X_onp[0])
        if self.order_flag:
            idx, X, Y = order_observations_by_grid(self.X_onp[0], self.Y_onp[0])
            self.order_indexes["grid_sort_idx"] = idx
        else:
            X, Y = reshape_by_grid(self.X_onp[0], self.Y_onp[0])

        self.X_onp = [X]
        self.X_arr = [np.array(X)]

        if Y is not None:
            self.Y_onp = [Y]
            self.Y_arr = [np.array(Y)]

        if Y is not None:
            self.create_mask()

    def get_meta(self):
        X = self.X_onp[0]

        dt = np.concatenate([np.array([0.0]), onp.diff(onp.unique(X[..., 0]))])

        self.meta["dt"] = dt
        self.meta["N"] = self.X_onp[0].shape[0]

    def get_spatial_locations(self):
        return self.X_arr[0][0, ...]

    def get_temporal_locations(self):
        return [self.meta["time_points"]]
