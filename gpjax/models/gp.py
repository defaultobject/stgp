from . import Model
from .. import Parameter
from .. import Kernel
from .. import Likelihood
from .. import Inference
from .. import Sparsity

from ..approximate_posteriors import *

from ..inference import *
from ..computation import *

from ..computation.marginal_likelihoods import gaussian_gaussian_marginal_likelihood
from ..computation.predictors import gp_predict_y, gp_predict_y_diagional
from ..decorators import return_gradients
from ..distributions import *

from .. import Dispatcher

from . import SDE_GP, VGP, CVI_VGP, BatchGP, CVI_SDE_VGP, ST_SDE_GP, ST_CVI_SDE_VGP

import jax.numpy as np
from jax import jit, partial

import typing 
from typing import Union, List, Optional

import inspect 

class GP(Model):
    def __init__(
        self, 
        X: Union[np.ndarray, List[np.ndarray]], 
        Y: Union[np.ndarray, List[np.ndarray]], 
        kernel: Optional[Kernel]=None, 
        likelihood: Optional[Likelihood]=None,
        inference: Optional[Inference]=None,
        sparsity: Optional[Sparsity] = None,
        options: Optional[dict] = None,
        name: Optional[str] = None
    ) -> None:
        super(GP, self).__init__(X, Y, kernel, likelihood, inference,  sparsity, options, name)

        self.base_model = None
        self.get_model()

    def get_model(self):
        #We match models based on the follow:
        #    type of inference i.e Batch, VI, StateSpace etc
        #    if VI then we further match bases on the approximate posterior
        #    the input dimensions of the data 
        #get the class that matched the type

        input_dim = self.X[0].shape[1]
        
        #if approximate posterior exists match class based on this

        gp_class = Dispatcher.dispatch('gp_model', type(self.inference), None, input_dim)
        if gp_class is False:
            #find VI models
            gp_class = Dispatcher.dispatch('gp_model', type(self.inference), type(self.inference.variational_posterior), input_dim)
            if gp_class is False:
                raise NotImplementedError()


        self.base_model = gp_class(self.X, self.Y, self.kernel, self.likelihood, self.inference, self.sparsity,  self.options)

        #copy the methods from self.model to this class
        self.add_properties()

    def add_properties(self):
        #hacky way to add all the methods in self.model that are not in this class
        # to this class

        #go through every property of self.model
        for p in dir(self.base_model):
            #check if p is a method
            if inspect.ismethod(getattr(self.base_model, p)):
                #check that self does not have this method
                if not hasattr(self, p):
                    setattr(self, p, getattr(self.base_model, p))

    @return_gradients
    def get_objective(self):
        return self.base_model.get_objective(return_grad=False, jit=False)

    def predict_y(self, data, diagonal_var: Optional[bool]=True):
        return self.base_model.predict_y(data, diagonal_var=diagonal_var)

    def predict_f(self, data, diagonal_var: Optional[bool]=True):
        return self.base_model.predict_f(data, diagonal_var=diagonal_var)



