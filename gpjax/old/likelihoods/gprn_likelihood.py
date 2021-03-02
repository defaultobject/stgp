from . import Likelihood
from ..parameter import Parameter
from ..computation import *
from ..computation.gaussian import log_gaussian_scalar

from ..data import Data, ListData

from ..decorators import return_gradients


import jax
import jax.numpy as np
from jax.experimental import loops
import numpy as onp

import typing
from typing import List, Optional


class GPRN_Likelihood(Likelihood):
    def __init__(
        self,
        num_outputs: Optional[int] = 1,
        num_latents: Optional[int] = 1,
        variances: Optional[np.ndarray] = None,
        trainable: Optional[bool] = True,
    ):
        self.name = "GPRN_Likelihood"

        self.num_outputs = num_outputs
        self.num_latents = num_latents

        self.P = self.num_outputs
        self.Q = self.num_latents

        super(GPRN_Likelihood, self).__init__(
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
        # Q latent functions
        F_arr = F_mu[: self.num_latents]
        # PQ latent functions - Row-major ordering
        # P descripts the columns, Q describes the Rows
        W_arr = F_mu[self.num_latents :]
        return F_arr, W_arr

    def log_likelihood(self, data: ListData, F_mu: List[np.ndarray]):
        Y = data.Y

        variance = self.variance
        mu_arr = self.conditional_mean(F_mu)

        mask = data.mask

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
        F_mu in D x N x 1
        """
        F_arr, W_arr = self.get_f_w(F_mu)
        mu_arr = []
        for p in range(self.P):
            mu = 0.0
            for q in range(self.Q):
                row_major_idx = p * self.Q + q
                W_pq = W_arr[row_major_idx]
                F_q = F_arr[q]
                F_q = W_pq * F_q
                mu += F_q

            mu_arr.append(mu)

        return mu_arr

    def conditional_var(self):
        return self.variance
