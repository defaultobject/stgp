from .data import Data

import jax.numpy as np
import numpy as onp

from typing import List


class TimeseriesData(Data):
    @property
    def X(self):
        # TODO generalise
        X = self.X_arr[0][self.order_indexes["argsort"], :]

        if len(X.shape) == 1:
            X = np.expand_dims(X, 0)

        return [X]

    @property
    def Y(self):
        Y = self.Y_arr[0][self.order_indexes["argsort"], :]
        return [Y]

    @property
    def mask(self):
        if self.Y_onp is None:
            mask = np.array(onp.full((self.X_onp[0].shape[0]), False, dtype=bool))
        else:
            mask = np.array(self.Y_mask[0][self.order_indexes["argsort"]])
        return mask

    @property
    def dt(self):
        return self.meta["dt"]

    @property
    def N(self):
        return self.meta["N"]

    def order(self):
        X = self.X_onp[0]

        sort_idx = onp.squeeze(onp.argsort(X, axis=0))
        self.order_indexes["argsort"] = sort_idx

    def get_meta(self):
        X = self.X_onp[0]
        dt = np.concatenate([np.array([0.0]), onp.diff(onp.unique(X[:, 0]))])

        self.meta["dt"] = dt
        self.meta["N"] = self.X_onp[0].shape[0]
