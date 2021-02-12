from . import Data

import jax.numpy as np
import numpy as onp

from typing import List

class PlaceholderData(Data):
    def __init__(self, X_arr: List[np.ndarray], Y_arr: List[np.ndarray], X_onp: List[onp.ndarray]=None, Y_onp: List[onp.ndarray]=None,  Y_mask: List[onp.ndarray]=None, order_indexes:dict = None, meta:dict = None, name='data'):

        super(PlaceholderData, self).__init__(X_arr, Y_arr, None, None)


