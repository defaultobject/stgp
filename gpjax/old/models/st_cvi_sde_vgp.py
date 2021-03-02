from . import Model, CVI_VGP
from .. import Parameter
from .. import Kernel
from .. import Likelihood
from .. import Inference
from .. import Sparsity

from ..data import Data, SpatioTemporalData

from . import ST_SDE_GP, BatchGP
from ..inference import *
from ..distributions import *
from ..decorators import return_gradients

from ..likelihoods import DiagonalGaussianLikelihood

from ..settings import Settings

from ..approximate_posteriors import BlockDiagonalConjugateApproxPosterior
from ..sparsity import NoSparsity

import jax.numpy as np
from jax import jit, partial

from .. import Dispatcher

import numpy as onp

import typing
from typing import Union, List, Optional

import warnings


@Dispatcher.register("gp_model", StateSpaceVI, None, None)
class ST_CVI_SDE_VGP(CVI_VGP):
    def setup_conjugate_model(self):
        """
        Setup spatio-temporal SDE-VGP
        """

        self.data = SpatioTemporalData(self.X, self.Y, self.X_onp, self.Y_onp)

        self.X_arr = self.data.flattened_X
        self.Y_arr = self.data.flattened_Y

        if type(self.sparsity) is NoSparsity:
            raise NotImplementedError()

        if not self.inference.is_initialized():
            # The optimal ApproximatePosterior is

            self.batch_inference = StateSpace(name="state_space")
            # self.conjugate_model = ST_SDE_GP(self.X, self.Y, inference=self.batch_inference, options=self.options, set_defaults=False, sparsity=self.sparsity)

            approx_posterior = BlockDiagonalConjugateApproxPosterior(
                data=self.data,
                sparsity=self.sparsity,
                conjugate_model=ST_SDE_GP,
                batch_inference=batch_inference,
                kernel=self.kernel,
            )

            self.inference.initialize(approx_posterior)

    def get_m_s(self, predict=True):
        q_p = self.inference.variational_posterior
        XS = self.data.X[0]
        return q_p.predict_f(
            self.data,
            self.data,
            self,
            0,
            diagonal_var=False,
            spatial_predict=predict,
            temporal_predict=False,
        )

    # @jit
    def get_ell_term_wrt_m_s(self, q_mean, q_var):
        q_p = self.inference.variational_posterior
        approx_data = q_p.approx_data

        mean, var = q_p.spatial_conditional(
            self.data, q_mean, q_var, self, 0, True, True, False
        )

        # E_q(f) [ log p( Y | f) ]
        ell_1 = q_p.precomputed_expected_log_likelihood(
            self.data, self.likelihood, mean, var, latent=0
        )

        return np.sum(ell_1)

    @return_gradients
    def get_objective(self):
        """
        See
           `Fast Variational Learning in State-Space Gaussian Process Models' - Chang et al

        The ELBO is

            E_q(f) [ log p( Y | f) ] - E_q(f) [ log N( m | f, s ) ] + log N( m | s)

        """
        q_p = self.inference.variational_posterior
        approx_data = q_p.approx_data

        # get q(u) = N(u|m, S)
        m, s = self.get_m_s(predict=False)

        # get q(f) = \int p(f|u) q(u) du
        f_mean, f_var_diag = q_p.spatial_conditional(
            self.data, m, s, self, latent=0, spatial_predict=True, diagonal_var=True
        )

        u_mean, u_var = q_p.spatial_conditional(
            approx_data.X[0],
            m,
            s,
            self,
            latent=0,
            spatial_predict=False,
            diagonal_var=False,
        )

        # E_q(f) [ log p( Y | f) ]
        ell_1 = q_p.precomputed_expected_log_likelihood(
            self.data, self.likelihood, f_mean, f_var_diag, latent=0
        )

        # E_q(f) [ log N( m | f, s ) ]
        approx_lik = q_p.distribution
        m = approx_lik.Y

        ell_2 = q_p.precomputed_expected_log_likelihood(
            approx_data, q_p.distribution, u_mean, u_var, latent=0
        )

        # log N( m | s)
        marginal_likelihood = q_p.get_approx_marginal_likelihood(self)

        debug = False
        if debug:
            print("ell_1: ", ell_1)
            print("kl: ", ell_2 + marginal_likelihood)
            print("Terms : ", ell_1, ell_2, marginal_likelihood)
            exit()

        elbo = ell_1 - ell_2 + marginal_likelihood

        return -np.squeeze(elbo)
