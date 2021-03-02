from . import Model
from .. import Parameter
from .. import Kernel
from .. import Likelihood
from .. import Inference
from .. import Sparsity
from .. import Data

from ..inference import *
from ..computation import *
from ..likelihoods import LMC_Likelihood

from ..computation.marginal_likelihoods import gaussian_lmc_marginal_likelihood
from ..computation.predictors import lmc_predict_y, lmc_predict_y_diagional
from ..decorators import return_gradients
from ..distributions import *

from ..data import ListData

from . import GP
from . import VLMC

import jax.numpy as np
from jax import jit, partial

import typing
from typing import Union, List, Optional


class LMC(GP):
    def __init__(
        self,
        X: Union[np.ndarray, List[np.ndarray]],
        Y: Union[np.ndarray, List[np.ndarray]],
        kernel: Optional[Kernel] = None,
        likelihood: Optional[Likelihood] = None,
        inference: Optional[Inference] = None,
        sparsity: Optional[Sparsity] = None,
        options: Optional[dict] = None,
        name: Optional[str] = None,
        num_latents: Optional[int] = 1,
    ) -> None:

        self.save_inputs_to_properties(locals())

        super(LMC, self).__init__(
            self.X,
            self.Y,
            self.kernel,
            self.likelihood,
            self.inference,
            self.sparsity,
            self.options,
            self.name,
        )

    def set_defaults_if_needed(self):
        # TODO(ollie): this should be done on the standarized inputs to avoid the if statement
        if type(self.Y) is list:
            self.num_outputs = len(self.Y)
        else:
            self.num_outputs = self.Y.shape[1]

        if self.likelihood is None:
            self.likelihood = LMC_Likelihood(
                num_outputs=self.num_outputs, num_latents=self.num_latents
            )

        super(LMC, self).set_defaults_if_needed()

    def get_model_dicts(self) -> dict:
        return {BatchInference: BatchLMC, VariationalInference: VLMC}

    def get_model(self):
        model_dicts = self.get_model_dicts()
        for key, func in model_dicts.items():
            if isinstance(self.inference, key):
                self.base_model = model_dicts[key](
                    self.X,
                    self.Y,
                    self.kernel,
                    self.likelihood,
                    self.inference,
                    self.sparsity,
                    self.options,
                    self.name,
                    self.num_latents,
                )
                return

        raise NotImplementedError(
            "LMC with inference {inf} has not been implemented yet".format(
                inf=self.inference
            )
        )


class BatchLMC(Model):
    def __init__(
        self,
        X: Union[np.ndarray, List[np.ndarray]],
        Y: Union[np.ndarray, List[np.ndarray]],
        kernel: Optional[Kernel] = None,
        likelihood: Optional[Likelihood] = None,
        inference: Optional[Inference] = None,
        sparsity: Optional[Sparsity] = None,
        options: Optional[dict] = None,
        name: Optional[str] = None,
        num_latents: Optional[int] = 1,
    ) -> None:
        self.save_inputs_to_properties(locals())
        super(BatchLMC, self).__init__(
            X, Y, self.kernel, self.likelihood, inference, sparsity, options, name
        )

        self.data = ListData(self.X, self.Y, self.X_onp, self.Y_onp)

    def setup(self):
        self.prior_arr = []
        for latent in range(self.num_latents):
            # Every output must have the same X
            prior = KernelGaussianDistribution(
                kernel=self.kernel_arr[latent], meta={"X": self.X_arr[0]}
            )
            self.prior_arr.append(prior)

    @return_gradients
    def get_objective(self):
        ml = gaussian_lmc_marginal_likelihood(
            self.data, self.likelihood, self.prior_arr
        )
        return -ml

    def predict_y(self, XS: np.ndarray, diagonal_var: bool):
        mu, sig = lmc_predict_y(XS, self.data, self.likelihood, self.prior_arr)

        if diagonal_var is False:
            return mu, sig

        # return diagional
        N = XS.shape[0]

        # extract diagional
        mu, var = mu, np.diag(sig)[:, None]

        # convert block form to list over outputs
        num_outputs = self.likelihood.num_outputs
        mu = [mu[p * N : N * (p + 1), :] for p in range(num_outputs)]
        var = [var[p * N : N * (p + 1), :] for p in range(num_outputs)]

        return mu, var

    def predict_f(self, XS: np.ndarray, diagonal_var: bool):
        raise NotImplementedError("Predict f is not implemented yet")
