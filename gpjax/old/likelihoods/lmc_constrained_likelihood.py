from . import Likelihood
from ..parameter import Parameter
from ..distributions import GaussianDistribution
from ..computation import *
from ..computation.gaussian import log_gaussian_scalar
from ..computation.general import get_correlation_cholesky, correlation_transform
from ..computation.general import inv_positive_transform
from ..settings import Settings
import warnings

from ..decorators import return_gradients

import jax
import jax.numpy as np
from jax import jit

from jax.experimental import loops
import numpy as onp

import typing
from typing import Union, List, Optional


class LMC_Constrained_Likelihood(Likelihood):
    def __init__(
        self,
        num_outputs: Optional[int] = 1,
        num_latents: Optional[int] = 1,
        variances: Optional[np.ndarray] = None,
        mixing_weights: Optional[np.ndarray] = None,
        trainable: Optional[bool] = True,
    ):
        self.name = "LMC_Constrained_Likelihood"

        self.num_outputs = num_outputs
        self.num_latents = num_outputs

        super(LMC_Constrained_Likelihood, self).__init__(
            name=self.name, meta={}, trainable=trainable
        )

        if variances is not None:
            if variances.shape[0] > 1 and variances.shape[0] != num_outputs:
                raise RuntimeError(
                    self.name,
                    ": Number of variances {var_num} should match number of outputs {num_outputs}".format(
                        var_num=variances.shape[0], num_outputs=num_outputs
                    ),
                )
        else:
            if Settings.strict_mode:
                raise RuntimeError(
                    "LMC_Constrained_Likelihood variance is not initalised"
                )

            warnings.warn(
                "LMC_Constrained_Likelihood variance is not initalised. Default will be used."
            )
            variances = (
                [inv_positive_transform(2.0) for output in range(num_outputs)]
                if variances is None
                else variances
            )
            variances = np.array(variances)

        self.variances = self.parameter(
            val=variances,
            constraint="positive",
            train=trainable,
            module_name=self.name,
            param_name="variance",
        )

        self.P = self.num_outputs
        self.Q = int(self.P * (self.P - 1) / 2)

        if mixing_weights is None:
            mixing_weights = onp.random.randn(self.Q)

        # TODO(ollie) - make the correlation trainform value a parameter
        self.mixing_weights = self.parameter(
            val=mixing_weights,
            constraint=lambda x: correlation_transform(x, 1.0),
            train=trainable,
            module_name=self.name,
            param_name="mixing_weights",
        )

    @property
    def coregion_weights(self):
        z_arr = self.mixing_weights.val
        correlation_cholesky = get_correlation_cholesky(z_arr, self.P, self.Q)
        return correlation_cholesky

    @property
    def variance(self):
        return self.variances.val

    def log_likelihood(self, Y: List[np.ndarray], F_mu: List[np.ndarray]):
        """
        Y - P outputs
        F_mu - Q latent functions
        """
        mixing_weights = self.coregion_weights
        variance = self.variance

        log_lik = 0
        for p in range(self.P):
            output_weights = mixing_weights[p, :]
            mu = np.sum([output_weights[q] * F_mu[q] for q in range(self.Q)], axis=0)
            mu = mu.reshape([Y[p].shape[0], 1])

            ll_p = log_gaussian_scalar(Y[p], mu, variance[p])

            log_lik += ll_p

        return log_lik

    def predict_mean_var(self, latent_mu_arr, latent_var_arr):
        mu_arr = []
        sig_arr = []
        mixing_weights = self.coregion_weights
        num_latents = self.num_latents
        variance = self.variance

        for output in range(self.num_outputs):
            output_weights = mixing_weights[output, :]
            mu = np.sum(
                [output_weights[q] * latent_mu_arr[q] for q in range(num_latents)],
                axis=0,
            )
            sig = np.sum(
                [
                    (output_weights[q] ** 2) * latent_var_arr[q]
                    for q in range(num_latents)
                ],
                axis=0,
            )
            sig = sig + variance[output]
            sig = np.reshape(sig, [sig.shape[0], 1])

            mu_arr.append(mu)
            sig_arr.append(sig)

        return mu_arr, sig_arr
