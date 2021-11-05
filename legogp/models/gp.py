from abc import ABC, abstractmethod
import objax
import jax.numpy as np
import numpy as onp
from typing import Optional, Tuple

from ..decorators import strict_mode_check, ensure_data
from ..obj_dispatch import obj_dispatch, obj_find

from ..inference import Batch

import json

class Model(objax.Module, ABC):
    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self._parent = None
        if 'parent' in kwargs.keys():
            self._parent = kwargs['parent']

        self._child = None # will be set when the model tree is constructed

    @abstractmethod
    def set_defaults(self):
        pass

    @abstractmethod
    def fix_inputs(self):
        pass

    @abstractmethod
    def get_objective(self):
        pass

    def print(self):
        var_dict = self.vars()
        for k, v in var_dict.items():
            print(f'{k}: {v.shape}')

    def checkpoint(self, name='model_checkpoint'):
        objax.io.save_var_collection(f'{name}.npz', self.vars())

    def load_from_checkpoint(self, name='model_checkpoint'):
        objax.io.load_var_collection(f'{name}.npz', self.vars())

def GP(*args, inference=Batch(), **kwargs):
    return obj_find('Model', inference)(*args, **kwargs)
