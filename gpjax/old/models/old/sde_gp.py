from . import Model

from ..distributions import (
    GaussianDistribution,
    ZeroMeanGaussianDistribution,
    KernelGaussianDistribution,
    DiagionalGaussianDistribution,
    SparseGaussianDistribution,
    WhitenedGaussianDistribution,
)
from ..likelihoods import GaussianLikelihood
from ..kernels import RBF, Matern32
from ..decorators import return_gradients
from .. import Parameter
from ..settings import Settings
from ..computation.general import (
    get_computational_primitives,
    kf_cholesky_solve_trace,
    cholesky_solve,
)
from ..computation.kalman_filter import kalman_loop, kalman_loop_store_intermediate
from ..computation.rts_smoother import rts_smoother

import jax
import jax.numpy as np
import numpy as onp
from jax import value_and_grad
from jax import jit, partial

from jax.config import config

config.update("jax_enable_x64", True)

import json

import matplotlib.pyplot as plt


class SDE_GP(Model):
    def __init__(self, X, Y, params=None, num_latents=None):
        """
        @TODO: default behaviours:
            if Y in N x P then have P independent latent functions
            if have num_latents and P = 1 then just have P versions of the same GP
            if have num_latents != P && P > 1 then throw run time error
        """
        if params == None:
            params = {}

        self.X = np.array(X)
        self.Y = np.array(Y)

        self.N = Y.shape[0]

        self.likelihood = GaussianLikelihood(params={"variance": np.log(1.0)})

        self.kernel = Matern32(
            params={"lengthscale": np.log(1.0), "variance": np.log(1.0)},
            name="Matern32_1",
        )

        prior = KernelGaussianDistribution(meta={"kernel": self.kernel})

        self.setup()

    def setup(self):
        # only get the difference on the time axis
        self.dt = np.concatenate([np.array([0.0]), onp.diff(self.X[:, 0])])

        # mask for nans in Y
        mask = onp.zeros(self.Y.shape[0], dtype=bool)
        mask[onp.argwhere(onp.isnan(self.Y)[:, 0])] = True
        self.mask = np.array(mask)

        self.no_mask = onp.zeros(self.Y.shape[0], dtype=bool)
        self.no_mask = np.array(self.no_mask)

    @return_gradients
    def get_objective(self):
        neg_log_marg_lik, _, _, _, _, _ = kalman_loop(
            self.Y, self.N, self.dt, self.kernel, self.likelihood, self.mask
        )

        return neg_log_marg_lik

    def predict(self, X):
        (
            neg_log_marg_lik,
            filtered_mean,
            filtered_cov,
            alpha,
            beta,
            chol_diag,
        ) = kalman_loop_store_intermediate(
            self.Y, self.N, self.dt, self.kernel, self.likelihood, self.mask
        )

        mu, sig, alpha = rts_smoother(
            self.Y,
            self.N,
            self.dt,
            filtered_mean,
            filtered_cov,
            self.kernel,
            self.likelihood,
            self.mask,
            alpha,
        )

        return mu, sig

    def get_computational_primitives(self, X, Y, L_sqrt):
        print("==========HERE=======")
        N = self.N

        # For some reason when using the uncommented lines below it becomes very slow. Does jit partial work on object id not values?
        # dt = np.concatenate([np.array([0.0]), onp.diff(X[:, 0])])
        # ASSUME NO NANS
        # mask = onp.zeros(Y.shape[0], dtype=bool)
        # mask = np.array(mask)

        lik_var = 1e-8  # jitter
        likelihood = GaussianLikelihood(
            params={"variance": np.log(lik_var)}, trainable=False
        )

        alpha, beta, chol_diag = get_computational_primitives(
            Y, N, self.dt, self.kernel, self.likelihood, self.no_mask
        )

        likelihood = GaussianLikelihood(
            params={"variance": np.log(0.0)}, trainable=False
        )
        A_trace = kf_cholesky_solve_trace(
            L_sqrt, N, self.dt, self.kernel, likelihood, self.no_mask
        )

        return alpha, beta, chol_diag, A_trace
