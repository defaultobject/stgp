from .. import Parameter
from .. import Module

import jax.numpy as np
from jax import jit, partial

from abc import ABC
from abc import abstractmethod


class Inference(Module):
    def __init__(self, name):
        super(Inference, self).__init__(name)

    def get_objective(self):
        pass
