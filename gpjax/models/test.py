from .. import Parameter
from .. import Kernel
from .. import Likelihood
from .. import Sparsity

from ..kernels import RBF
from ..likelihoods import GaussianLikelihood
from ..inference import BatchInference
from ..module import Module
from ..sparsity import NoSparsity

from ..decorators import return_gradients, set_defaults_from_self, ensure_data_passed

from ..data import Data, ListData

import typing 
from typing import Union, List, Optional

import jax.numpy as np
import jax
from jax import jit, partial

import numpy as onp

from abc import ABC
from abc import abstractmethod

import warnings

import json
import pickle

from ..settings import Settings


class TestModel(Module):
    def __init__(
        self, 
        X: Union[onp.ndarray, np.ndarray, List[np.ndarray]]=None, 
        Y: Optional[Union[onp.ndarray, np.ndarray, List[np.ndarray]]]=None, 
        kernel: Optional[Kernel]=None, 
        likelihood: Optional[Likelihood]=None,
        inference: Optional['Inference']=None,
        sparsity: Optional[Sparsity] = None,
        options: Optional[dict] = None,
        name: Optional[str] = None,
        X_onp: Optional[onp.ndarray]=None,
        Y_onp: Optional[onp.ndarray]=None,
        set_defaults:Optional[bool] = True,
        key=None
    ) -> None:

        self.save_inputs_to_properties(locals())

        if self.likelihood is None:
            self.likelihood = GaussianLikelihood() 

        if name is None:
            self.name = 'TestModel'

        super(TestModel, self).__init__(name=self.name)

    @return_gradients
    #@set_defaults_from_self
    @jit
    def get_objective(self, data:Optional[Data] = None):

        #key = list(Parameter.PARAM_DICT.keys())[0]
        #return 2* Parameter.PARAM_DICT[key]
        return 2.0*np.sum(np.exp(self.likelihood.variance))

