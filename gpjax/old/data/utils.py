from . import TimeseriesData, SpatioTemporalData, ST_PlaceholderData

import numpy as onp

import jax
import jax.numpy as np


def order_timeseries(data_xs, data, latent=0):
    X_onp = data.X_onp[latent]

    X = data.X[latent]
    Y = data.Y[latent]

    XS = data_xs.X
    if type(XS) is list:
        XS = XS[0]

    # create nan observations for XS
    YS = onp.zeros([XS.shape[0], Y.shape[1]]) * onp.NaN

    # onp.unique returns the idices of the first occurence.
    # we want to use training data over testing, so place in front
    X_all = np.concatenate([X, XS], axis=0)
    Y_all = np.concatenate([Y, YS], axis=0)

    # generate indexes for test locations
    test_idx = onp.arange(X_onp.shape[0], X_onp.shape[0] + YS.shape[0])

    # X and XS may have duplicate time entries, we only want the unique ones
    # unique_reverse_idx maps from the unique only array to the original array
    X_unique, unique_idx, unique_reverse_idx = onp.unique(
        X_all, return_index=True, return_inverse=True
    )

    X_all = X_all[unique_idx, :]
    Y_all = Y_all[unique_idx, :]

    # get index that sorts X_all
    sort_idx = onp.argsort(X_all, axis=0)

    # sort X_all and Y_all
    X_all = X_all[sort_idx[:, 0], :]
    Y_all = Y_all[sort_idx[:, 0], :]

    order_indexes = {
        "sort_idx": sort_idx,
        "test_idx": test_idx,
        "unique_reverse_idx": unique_reverse_idx,
    }

    return X_all, Y_all, order_indexes


def order_timeseries_prediction(data_xs, data, latent=0):
    X_all, Y_all, order_indexes = order_timeseries(data_xs, data, latent=latent)
    data = TimeseriesData([X_all], [Y_all], [X_all], [Y_all])
    data.order_indexes.update(order_indexes)
    return data


def order_spatiotemporal_prediction(data_xs, data, latent=0):
    """
    This only supports predictions at new time locations / spatial slices not new spatial locations. That is handled by a separate conditional after prediction.
    """

    X_onp = data.X_onp[latent]

    # Ordered data?
    X = data.X[latent]
    Y = data.Y[latent]

    num_spatial_locations = X.shape[1]

    XS = data_xs.X[0]

    Nt = XS.shape[0]
    Ns = XS.shape[1]
    D = XS.shape[2]
    N = Nt * Ns

    # merge test and training time points
    xs_timeseries = data_xs.get_temporal_locations()[0]
    x_timeseries = data.get_temporal_locations()[0]

    xs_timeseries = np.reshape(xs_timeseries, [-1, 1])
    x_timeseries = np.reshape(x_timeseries, [-1, 1])

    Y_timeseries = Y[..., 0]

    xs_timeseries = TimeseriesData([xs_timeseries], None, [xs_timeseries], None)
    x_timeseries = TimeseriesData(
        [x_timeseries], [Y_timeseries], [x_timeseries], [Y_timeseries]
    )

    X_all, Y_all, order_indexes = order_timeseries(xs_timeseries, x_timeseries)

    # Data must have the same spatial points at every time location. We jsut tile these points across all test+trainig locations
    X_space = X[0, ...]

    X_all = np.expand_dims(X_all, -1)
    X_all = np.tile(X_all, (1, X_space.shape[0], 1))

    def concat(time, Z):
        return np.hstack([time, Z[:, 1:]])

    X_all = jax.vmap(concat, (0, None), 0)(X_all, X_space)

    Nt_all = X_all.shape[0]
    Ns_all = X_all.shape[1]
    N_all = Nt_all * Ns_all

    X_all = np.reshape(X_all, [N_all, D])

    Y_all = np.reshape(Y_all, [N_all, 1])

    data = SpatioTemporalData([X_all], [Y_all], [X_all], [Y_all], order_flag=False)

    data.order_indexes.update(order_indexes)
    return data


def unorder_timeseries_prediction(data_all, mu, sig):
    unique_reverse_idx = data_all.order_indexes["unique_reverse_idx"]
    test_idx = data_all.order_indexes["test_idx"]
    sort_idx = data_all.order_indexes["sort_idx"]

    mu = mu[unique_reverse_idx, :][test_idx, :]
    sig = sig[unique_reverse_idx, :][test_idx, :]

    mu = np.squeeze(mu)
    sig = np.squeeze(sig)

    return mu, sig


def unorder_spatiotemporal_prediction(data_all, mu, sig):
    return unorder_timeseries_prediction(data_all, mu, sig)
