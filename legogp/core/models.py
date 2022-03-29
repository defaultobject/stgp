"""
Defines the abstract model classes.
Models are split into priors (no Y) and posteriors (has Y)
"""
from abc import ABC, abstractmethod
import objax
import jax.numpy as np
import numpy as onp
from typing import Optional, Tuple
from ..utils import utils
from ..data import Data

class Model(objax.Module, ABC):
    def __init__(self, **kwargs):
        super(Model, self).__init__()

    def set_defaults(self):
        pass

    def fix_inputs(self):
        pass

    @property
    def name(self) -> int:
        return 'model_checkpoint'

    def input_dim(self) -> int:
        raise NotImplementedError()

    def output_dim(self) -> int:
        raise NotImplementedError()

    def mean(self, XS: np.ndarray) -> np.ndarray :
        raise NotImplementedError()

    def var(self, XS: np.ndarray) -> np.ndarray :
        raise NotImplementedError()

    def covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray :
        raise NotImplementedError()

    def print(self):
        param_dict = utils.get_parameters(self)

        for k, v in param_dict.items():
            print(f'{k}: {v}')

    def checkpoint(self, name=None):
        if name is None:
            name = self.name

        objax.io.save_var_collection(f'{name}.npz', self.vars())

    def load_from_checkpoint(self, name=None):
        if name is None:
            name = self.name

        objax.io.load_var_collection(f'{name}.npz', self.vars())

    def get_fixed_params(self):
        return utils.get_fixed_params(self)

class Prior(Model):
    @property
    #@abstractmethod
    def kernel(self) -> 'Kernel':
        pass

    @property
    #@abstractmethod
    def mean(self, XS) -> 'Kernel':
        pass

    @property
    #@abstractmethod
    def transform(self) -> 'Kernel':
        pass

class Posterior(Model):
    def __init__(self, X=None, Y=None, data=None, latent_y=False, latent_x=False, **kwargs):
        if X is not None and Y is not None:
            if data is None:
                data = Data(X, Y)

        self.data = data

    @property
    def X(self):
        raise NotImplementedError()

    @property
    def Y(self):
        raise NotImplementedError()

    def log_marginal_likelihood(self, X: Optional[np.ndarray] = None, Y: Optional[np.ndarray] = None):
        raise NotImplementedError()

    @property
    def prior(self):
        raise NotImplementedError()

    @property
    def likelihood(self):
        raise NotImplementedError()

    def predict_f(self, XS: np.ndarray, *args, **kwargs):
        raise NotImplementedError()

    def predict_y(self, XS: np.ndarray, *args, **kwargs):
        raise NotImplementedError()

    def posterior_blocks(self, *args, **kwargs):
        raise NotImplementedError()

    def posterior(self, *args, **kwargs):
        raise NotImplementedError()

