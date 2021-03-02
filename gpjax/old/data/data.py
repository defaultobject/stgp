from ..module import Module

import jax.numpy as np
import numpy as onp

from typing import List


class Data(Module):
    def __init__(
        self,
        X_arr: List[np.ndarray],
        Y_arr: List[np.ndarray],
        X_onp: List[onp.ndarray],
        Y_onp: List[onp.ndarray],
        Y_mask: List[onp.ndarray] = None,
        order_indexes: dict = None,
        meta: dict = None,
        name="data",
        order_flag=True,
    ):
        self.X_arr = X_arr
        self.Y_arr = Y_arr

        self.X_onp = X_onp
        self.Y_onp = Y_onp

        self.order_flag = order_flag

        if (
            (self.Y_onp is not None)
            and (self.Y_onp[0] is not None)
            and (Y_mask is None)
        ):
            self.create_mask()
        else:
            self.Y_mask = Y_mask

        self.name = name

        if meta is None:
            self.meta = {}

        if order_indexes is None and self.X_onp is not None:
            self.order_indexes = {}
            self.order()
        else:
            self.order_indexes = order_indexes

        if meta is None and self.X_onp is not None:
            self.get_meta()
        else:
            self.meta = meta

        super(Data, self).__init__(name=self.name)

    @property
    def X(self):
        return self.X_arr

    @property
    def Y(self):
        return self.Y_arr

    @property
    def mask(self):
        return self.Y_mask

    def create_mask(self):
        self.Y_mask = [
            onp.isnan(onp.squeeze(self.Y_onp[p])) for p in range(len(self.Y_onp))
        ]

    def order(self):
        pass

    def get_meta(self):
        pass
