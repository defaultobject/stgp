from . import Likelihood
from ..parameter import Parameter
from ..computation import *
from ..computation.gaussian import log_gaussian_scalar

from ..decorators import return_gradients

from ..data import Data, ListData


import jax
import jax.numpy as np
from jax.experimental import loops
import numpy as onp

import numpy as onp

import typing
from typing import List, Optional


class Constrained_GPRN_Likelihood(Likelihood):
    def __init__(
        self,
        num_outputs: Optional[int] = 1,
        num_latents: Optional[int] = 1,
        variances: Optional[np.ndarray] = None,
        trainable: Optional[bool] = True,
    ):
        self.name = "Constrained_GPRN_Likelihood"

        self.num_outputs = num_outputs
        self.num_latents = num_latents

        self.P = self.num_outputs
        # self.Q = self.num_latents
        self.Q = int(self.P * (self.P - 1) / 2)

        super(Constrained_GPRN_Likelihood, self).__init__(
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
            variances = (
                [1.0 for output in range(num_outputs)]
                if variances is None
                else variances
            )
            variances = np.array(variances)

        scope = "hyperparameter"
        self.variances = self.parameter(
            val=variances,
            constraint="positive",
            scope=scope,
            train=trainable,
            module_name=self.name,
            param_name="variance",
        )

    @property
    def variance(self):
        return self.variances.val

    def get_f_w(self, F_mu: List[np.ndarray]):
        """
        F_mu is an array of predicted means from each of the latent functions.
        Let T denote the total number of latent functions then

            F_mu in T x N x 1

        F_mu is ordered such that the first P latent functions describe F and the remainder
            describe the latent functions inside the coregionalilsation matrix.

        There are P latent functions F and (P-1)^2/2 latent functions \delta inside the coregionilisation matrix.

        To construct the coregionilisation matrix \delta, for each n, \delta_n is pushed through the cholesky correlation mapping which gives an N x P x P matrix. Ie for each location we have an P x P cholesky correlation matrix.

        """
        # P latent functions
        F_arr = F_mu[: self.P]

        # (P-1)^2/2 latent functions
        W_arr = F_mu[self.P :]

        constraint = lambda x: correlation_transform(x, 1.0)

        # TODO: assuming same sized latent functions
        N = F_arr[0].shape[0]

        # (P-1)(p-1)/2 x N x 1
        Z_arr = [constraint(w)[:, 0] for w in W_arr]

        correlation_vmap = jax.vmap(
            get_correlation_cholesky, in_axes=(0, None, None), out_axes=(0)
        )
        # Construct correlation matrix for each location
        correlation_matrix = correlation_vmap(Z_arr, self.P, self.Q)  # N x P x P

        # Transform correlation matrix into row major order
        W_arr = correlation_matrix.reshape(
            [N, self.P ** 2, 1], order="C"
        )  # N x P^2 x 1
        W_arr = np.transpose(W_arr, (1, 0, 2))

        return F_arr, W_arr

    def log_likelihood(self, data: ListData, F_mu: List[np.ndarray]):
        Y = data.Y

        variance = self.variance

        mu_arr = self.conditional_mean(F_mu)

        log_lik = 0
        for p in range(self.P):
            mu = mu_arr[p]

            mu = mu.reshape([Y[p].shape[0], 1])

            Y_p = Y[p]

            bool_mask = np.logical_not(data.mask[p])
            Y_p = Y_p[bool_mask, ...]
            mu = mu[bool_mask, ...]

            ll_p = log_gaussian_scalar(Y_p, mu, variance[p])

            log_lik += ll_p

        return log_lik

    def conditional_mean(self, F_mu: List[np.ndarray]):
        """
        Calculates:
            Y = W F
        """
        F_arr, W_arr = self.get_f_w(F_mu)
        mu_arr = []
        for p in range(self.P):
            mu = 0.0
            for q in range(self.P):
                row_major_idx = p * self.P + q
                W_pq = W_arr[row_major_idx]
                F_q = F_arr[q]
                F_q = W_pq * F_q
                mu += F_q

            mu_arr.append(mu)

        return mu_arr

    def conditional_var(self):
        return self.variance
