from . import Model
from .. import Parameter
from .. import Kernel
from .. import Likelihood
from .. import Inference
from .. import Sparsity

from ..data import Data, ListData

from ..distributions import *
from ..decorators import *

from ..settings import Settings

from ..approximate_posteriors import GaussianApproxPosterior, MeanFieldApproxPosterior

import jax.numpy as np
from jax import jit, partial

import numpy as onp

import typing
from typing import Union, List, Optional

import warnings


class VLMC(Model):
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
        super(VLMC, self).__init__(
            X, Y, self.kernel, self.likelihood, inference, sparsity, options, name
        )

        self.data = ListData(self.X, self.Y, self.X_onp, self.Y_onp)

    def setup(self):
        N = self.Y_arr[0].shape[0]
        self.N_latents = [N for y in range(self.num_latents)]

        # =============================setup approximate posterior=============================
        if not self.inference.is_initialized():
            if Settings.strict_mode:
                raise RuntimeError("Variational posterior is not initalised")

            warnings.warn(
                "No variational approximate posterior specified. Default will be used."
            )
            q = MeanFieldApproxPosterior()
            self.inference.initialize(q)

        self.inference.variational_posterior.setup(self.N_latents)

        # =============================setup prior=============================
        self.prior_arr = []
        for p in range(self.num_latents):
            if self.options["whiten"]:
                prior_p = WhitenedKernelGaussianDistribution(
                    kernel=self.kernel_arr[p], meta={"X": self.X_arr[p]}
                )
                self.prior_arr.append(prior_p)
            else:
                prior_p = KernelGaussianDistribution(
                    kernel=self.kernel_arr[p], meta={"X": self.X_arr[p]}
                )
                self.prior_arr.append(prior_p)

    @return_gradients
    def get_objective(self):
        ell = self.inference.variational_posterior.expected_log_likelihood(
            self.data, self.likelihood, self
        )
        kl = self.inference.variational_posterior.KL(self.X_arr, self.prior_arr)
        elbo = ell - kl

        return -elbo

    @ensure_data_passed
    def predict_y(self, data_xs, diagonal_var: bool):
        if diagonal_var:
            return self.inference.variational_posterior.predict_y(
                data_xs, self.data, self, diagonal_var=True
            )

        raise NotImplementedError("")

    def predict_latents(self, XS: np.ndarray, diagonal_var: bool):
        raise NotImplementedError("Predict f is not implemented yet")

    @ensure_data_passed
    def predict_f(self, data_xs, diagonal_var: bool):
        if diagonal_var:
            return self.inference.variational_posterior.predict_f(
                data_xs, self.data, self, diagonal_var, predict=True
            )
