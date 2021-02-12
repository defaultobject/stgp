from . import Model, Constrained_VLMC
from .. import Parameter
from .. import Kernel
from .. import Likelihood
from .. import Inference
from .. import Sparsity
from .. import Data

from ..inference import *
from ..computation import *
from ..likelihoods import LMC_Constrained_Likelihood

from ..computation.marginal_likelihoods import gaussian_lmc_marginal_likelihood
from ..computation.predictors import lmc_predict_y, lmc_predict_y_diagional
from ..decorators import return_gradients
from ..distributions import *

from ..data import ListData

from . import GP

import jax.numpy as np
from jax import jit, partial

import typing 
from typing import Union, List, Optional

class Constrained_LMC(GP):
    def __init__(
        self, 
        X: Union[np.ndarray, List[np.ndarray]], 
        Y: Union[np.ndarray, List[np.ndarray]], 
        kernel: Optional[Kernel]=None, 
        likelihood: Optional[Likelihood]=None,
        inference: Optional[Inference]=None,
        sparsity: Optional[Sparsity] = None,
        options: Optional[dict] = None,
        name: Optional[str] = None,
        num_latents: Optional[int]=1
    ) -> None:
        self.save_inputs_to_properties(locals())

        super(Constrained_LMC, self).__init__(self.X, self.Y, self.kernel, self.likelihood, self.inference, self.sparsity, self.options, self.name)

    def set_defaults_if_needed(self):
        #TODO(ollie): this should be done on the standarized inputs to avoid the if statement
        if type(self.Y) is list:
            self.num_outputs = len(self.Y)
        else:
            self.num_outputs = self.Y.shape[1]

        if self.likelihood is None:
            P = self.num_outputs
            Q = int((P-1)*P/2)

            mixing_weights = get_z_from_correlation_cholesky(np.eye(P), P, Q)
            mixing_weights = np.array([0.1, 1.0, 0.2])


            self.likelihood = LMC_Constrained_Likelihood(mixing_weights=None, num_outputs = self.num_outputs, num_latents=self.num_latents)

        super(Constrained_LMC, self).set_defaults_if_needed()

    def get_model_dicts(self) -> dict:
        return {
            BatchInference: Batch_Constrained_LMC,
            VariationalInference: Constrained_VLMC
        }

    def get_model(self):
        model_dicts = self.get_model_dicts()
        for key, func in  model_dicts.items():
            if isinstance(self.inference, key):
                self.model = model_dicts[key](self.X, self.Y, self.kernel, self.likelihood, self.inference, self.sparsity, self.options, self.name, self.num_latents)
                return

        raise NotImplementedError('GP with inference {inf} has not been implemented yet'.format(inf=self.inference))

class Batch_Constrained_LMC(Model):
    def __init__(
        self, 
        X: Union[np.ndarray, List[np.ndarray]], 
        Y: Union[np.ndarray, List[np.ndarray]], 
        kernel: Optional[Kernel]=None, 
        likelihood: Optional[Likelihood]=None,
        inference: Optional[Inference]=None,
        sparsity: Optional[Sparsity] = None,
        options: Optional[dict] = None,
        name: Optional[str] = None,
        num_latents: Optional[int]=1
    ) -> None:
        self.save_inputs_to_properties(locals())
        super(Batch_Constrained_LMC, self).__init__(X, Y, self.kernel, self.likelihood, inference, self.sparsity, options, name)

        self.data = ListData(self.X, self.Y, self.X_onp, self.Y_onp)

    def setup(self):
        self.prior_arr = []
        for latent in range(self.num_latents):
            prior = KernelGaussianDistribution(
                kernel=self.kernel_arr[latent],
                meta={
                    'X': self.X_arr[latent]
                }   
            ) 
            self.prior_arr.append(prior)

    @return_gradients
    @jit
    def get_objective(self):
        ml = gaussian_lmc_marginal_likelihood(self.data, self.likelihood, self.prior_arr)
        return -ml

    @partial(jit, static_argnums=(2))
    def predict_y(self, XS:np.ndarray, diagional_var: bool):
        mu, sig =  lmc_predict_y(XS, self.data, self.likelihood, self.prior_arr)

        if diagional_var is False:
            return mu, sig

        #return diagional
        N = XS.shape[0]

        num_outputs = self.likelihood.num_outputs
        mu, var = mu, np.diag(sig)[:, None]

        mu = [mu[p*N:N*(p+1), :] for p in range(num_outputs)]
        var = [var[p*N:N*(p+1), :] for p in range(num_outputs)]

        return mu, var

    @partial(jit, static_argnums=(2))
    def predict_f(self, XS:np.ndarray, diagional_var:bool):
        raise NotImplementedError('Predict f is not implemented yet')


