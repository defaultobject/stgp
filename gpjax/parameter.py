import jax
import jax.numpy as jnp
import numpy as np
import uuid
import random

import typing
from typing import List

from .computation.general import lower_triangle, positive_transform, negative_transform

from jax.numpy import vectorize

#TODO: this wont work if there are multiple kernels
#   FIX: need to jit the uuid that is originanly created

#monkey patch to create reproducible globally random numbers
#needed for checkpoints
rd = random.Random()
rd.seed(0)
uuid.uuid4 = lambda: uuid.UUID(int=rd.getrandbits(128))

class Parameter():
    #@TODO(ollie): is their a neater way to organise these dicts?
    #store all parameters
    ALL_PARAM_DICT = {}

    #store trainable parameters
    PARAM_DICT = {}

    #store all parameter objects
    OBJ_DICT = {}

    #store all parameters in their scope
    SCOPE_DICT = {}

    @staticmethod
    def get_params_in_scope(scope: str):
        if scope not in Parameter.SCOPE_DICT.keys():
            return []

        return Parameter.SCOPE_DICT[scope]

    @staticmethod
    def get_subset(param_names: List[str]):
        if scope not in Parameter.SCOPE_DICT.keys():
            return None

        return Parameter.SCOPE_DICT[scope]

    @staticmethod
    def clear():
        """
            Delete/reset all params
        """

        Parameter.ALL_PARAM_DICT = {}
        Parameter.OBJ_DICT = {}
        Parameter.PARAM_DICT = {}
        Parameter.SCOPE_DICT = {}
    
    @staticmethod
    def update(params: dict):
        """
            Update trainable paramaters with tracers and values in params
        """

        #overwrite trainable params with the updated params
        Parameter.PARAM_DICT = params

        for name in Parameter.PARAM_DICT.keys():
            Parameter.ALL_PARAM_DICT[name] = Parameter.PARAM_DICT[name]

            #Update the parameter objects and apply any constraints
            Parameter.OBJ_DICT[name].raw = Parameter.PARAM_DICT[name]
            Parameter.OBJ_DICT[name]._apply_constraint()

    def __init__(self, init=None, meta=None, shape=None, constraint=None, transform=None, name=None, scope=None, train=True):
        if (init is None) and (shape is None):
            raise NotImplementedError('Shape or Init must be specified')

        self.raw_name = name

        if name is None: 
            name = str(uuid.uuid4())
        else:
            name = name+'/'+str(uuid.uuid4())

        self.name = name

        self.meta = meta

        self.constraint = constraint

        if init is not None:
            self.raw = init

        if shape is not None and init is None:
            #only randomly asign if init is None
            self.raw = jnp.ones_like(shape)

        self._apply_constraint()

        if train:
            if name in Parameter.ALL_PARAM_DICT.keys():
                raise RuntimeError('{name} name already in Parameter dict'.format(name = name))

            Parameter.OBJ_DICT[name] = self
            Parameter.ALL_PARAM_DICT[name] = self.raw
            Parameter.PARAM_DICT[name] = self.raw

            if scope is not None:
                self.add_name_to_scope(name, scope)

    def add_name_to_scope(self, name: str, scope: str):
        if scope not in Parameter.SCOPE_DICT.keys():
            Parameter.SCOPE_DICT[scope] = []

        Parameter.SCOPE_DICT[scope].append(name)

    def _apply_constraint(self):
        """
            Apply the function to constrain self.value. If constraint does not exist, raise error.
        """
        if self.constraint is None:
            self.transformed_value = self.raw

        elif callable(self.constraint):
            self.transformed_value = self.constraint(self.raw)

        elif self.constraint == 'positive':
            self.transformed_value = positive_transform(self.raw)

        elif self.constraint == 'negative':
            self.transformed_value = negative_transform(self.raw)

        elif self.constraint == 'lower triangular':
            self.transformed_value = lower_triangle(self.raw, self.meta['N'])

        elif self.constraint == 'block lower triangular':
            self.transformed_value = jax.vmap(lower_triangle, in_axes=(0, None), out_axes=0)(self.raw, self.meta['N'])

        else:
            raise NotImplementedError('Constraint: {c} is not implemented'.format(c=self.constraint))

    def __call__(self):
        return self.value

    @property
    def val(self):
        #self._apply_constraint()
        return self.value

    @property
    def value(self):
        self._apply_constraint()
        return self.transformed_value

    @property
    def raw_value(self):
        return self.raw


def match_contains(arr, s):
    return [a for a in arr if s in a]
