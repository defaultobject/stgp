from .data import Data

import jax.numpy as np
import numpy as onp

from typing import List

class ListData(Data):
    @property
    def stacked_mask(self):
        mask = self.mask.copy()
        #each mask index starts form 0, need to shift each by the size of Y_arr[p]
        total_shift = 0
        for p in range(len(mask)):
            mask[p] = mask[p] + total_shift
            total_shift += mask[p].shape[0]

        return np.hstack(mask)


