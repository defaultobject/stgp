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
from .. import Parameter

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
    def __init__(self, X=None, Y=None, latent_y=False, latent_x=False, **kwargs):
        self._Y = Parameter(np.array(Y), train=latent_y, name='Y')
        self._X = Parameter(np.array(X), train=latent_x, name='X')

    @property
    def X(self):
        return self._X.value

    @property
    def Y(self):
        return self._Y.value

    #@abstractmethod
    def log_marginal_likelihood(self, X: Optional[np.ndarray] = None, Y: Optional[np.ndarray] = None):
        pass

    @property
    #@abstractmethod
    def prior(self):
        pass

    @property
    #@abstractmethod
    def likelihood(self):
        pass

    #@abstractmethod
    def predict_f(self, XS: np.ndarray, X: Optional[np.ndarray] = None, Y: Optional[np.ndarray] = None):
        pass

    #@abstractmethod
    def predict_y(self, XS: np.ndarray, X: Optional[np.ndarray] = None, Y: Optional[np.ndarray] = None):
        pass

