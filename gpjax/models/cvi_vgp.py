from . import Model
from .. import Parameter
from .. import Kernel
from .. import Likelihood
from .. import Inference
from .. import Sparsity

from ..data import Data, ListData

from . import BatchGP
from ..inference import *
from ..distributions import *
from ..decorators import *

from ..likelihoods import DiagonalGaussianLikelihood

from ..settings import Settings

from ..approximate_posteriors import DiagonalConjugateApproxPosterior
from ..sparsity import NoSparsity

from .. import Dispatcher

import jax.numpy as np
from jax import jit, partial

import numpy as onp

import typing 
from typing import Union, List, Optional

import warnings

#@Dispatcher.register('gp_model', VariationalInference, None, None)
class CVI_VGP(Model):
    def setup(self):
        """
            It is assumed that if the observations are passed as a list of input output arrays (e.g [X for i in num_latents])
                then this will create num_latent number of independent CVI_VGP objectives. The final objective is then just the sum of the 
                individual objectives.
        """


        self.N = self.Y_arr[0].shape[0]

        if type(self.sparsity) is not NoSparsity:
            self.N_latents = [s.Z.shape[0] for s in self.sparsity_arr]
        else:
            self.N_latents = [y.shape[0] for y in self.Y_arr]

        self.num_latents = len(self.X_arr)
        self.total_num_latents = self.num_latents

        #=============================setup approximate posterior=============================
        if not self.inference.is_initialized():
            if Settings.strict_mode:
                raise RuntimeError('Variational posterior is not initalised')

            warnings.warn('No variational approximate posterior specified. Default will be used.')
            #raise NotImplementedError()


        #self.inference.variational_posterior.setup(self.N_latents)

        #=============================setup prior=============================
        self.prior_arr = []
        for p in range(self.num_latents):

            if type(self.sparsity) is not NoSparsity:
                X = self.sparsity.Z[p]
            else:
                X = self.X_arr[p]

            if self.options['whiten'] == True:
                raise RuntimeError('Whiten must be false')

            if self.options['whiten']:
                prior_p = WhitenedKernelGaussianDistribution(
                    kernel=self.kernel_arr[p],
                    meta={
                        'X': X
                    }   
                )
                self.prior_arr.append(prior_p)
            else:
                prior_p = KernelGaussianDistribution(
                    kernel=self.kernel_arr[p],
                    meta={
                        'X': X
                    }   
                )
                self.prior_arr.append(prior_p)

        self.setup_conjugate_model()


    def setup_conjugate_model(self):

        if type(self.sparsity) is not NoSparsity:
            raise NotImplementedError('sparsity must be NoSparsity')

        if not self.inference.is_initialized():

            #construct a partial model to be setup on use
            self.batch_inference = BatchInference(name='batch_inference')

            approx_posterior = DiagonalConjugateApproxPosterior(data=self.data, sparsity=self.sparsity, conjugate_model=BatchGP, batch_inference=batch_inference, kernel=self.kernel)

            self.inference.initialize(approx_posterior)
        
    #@jit
    def get_m_s(self, predict=True):
        q_p = self.inference.variational_posterior
        XS = self.data.X[0]
        return q_p.predict_f(XS, self.data, self, 0, diagonal_var=True, predict=predict)


    #@jit
    def get_ell_term_wrt_m_s(self, q_mean, q_var):
        q_p = self.inference.variational_posterior
        approx_data = q_p.approx_data

        #E_q(f) [ log p( Y | f) ]
        ell_1 = q_p.precomputed_expected_log_likelihood(self.data, self.likelihood, q_mean, q_var, latent=0)
        return ell_1

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

        #get q(u) = N(u|m, S)
        m, s = self.get_m_s(predict=False)

        #get q(f) = \int p(f|u) q(u) du
        #f_mean, f_var_diag = q_p.spatial_conditional(self.data, m, s, self, latent=0, spatial_predict=True, diagonal_var=True)

        #u_mean, u_var = q_p.spatial_conditional(approx_data.X[0], m, s, self, latent=0, spatial_predict=False, diagonal_var=False)

        #E_q(f) [ log p( Y | f) ]
        ell_1 = q_p.expected_log_likelihood(self.data, self.likelihood, self, latent=0)
        #ell_1 = q_p.precomputed_expected_log_likelihood(self.data, self.likelihood, f_mean, f_var_diag, latent=0)

        #E_q(f) [ log N( m | f, s ) ]
        #m = q_p.lambda_1.value
        approx_lik = q_p.distribution
        m = approx_lik.Y

        ell_2 = q_p.expected_log_likelihood(approx_data, q_p.distribution, self, latent=0)
        #ell_2 = q_p.precomputed_expected_log_likelihood(approx_data, q_p.distribution, u_mean, u_var, latent=0)

        #log N( m | s)
        marginal_likelihood = q_p.get_approx_marginal_likelihood(self)

        elbo =  ell_1 - ell_2 + marginal_likelihood

        return -np.squeeze(elbo)

    #@partial(jit, static_argnums=(2))
    @ensure_data_passed
    @set_defaults_from_self
    def predict_y(self, data_xs:Data, data:Optional[Data] = None, diagonal_var: Optional[bool]=True):
        latent = 0
        q_p = self.inference.variational_posterior

        mu_p, pred_p = q_p.predict_y(data_xs, data, self, latent, diagonal_var)

        if False and diagional_var:
            #ensure correct sizes
            mu_p = np.reshape(mu_p, [mu_p.shape[0], 1])
            pred_p = np.reshape(pred_p, [pred_p.shape[0], 1])

        return [mu_p], [pred_p]
        #return self.batch_gp.predict_y(XS, diagional_var)

    @ensure_data_passed
    @set_defaults_from_self
    def predict_f(self, data_xs:Data, data:Optional[Data] = None, diagonal_var: Optional[bool]=True):
        latent = 0
        q_p = self.inference.variational_posterior

        mu_p, pred_p = q_p.predict_latent(data_xs, data, self, latent, diagonal_var)

        if False and diagional_var:
            #ensure correct sizes
            mu_p = np.reshape(mu_p, [mu_p.shape[0], 1])
            pred_p = np.reshape(pred_p, [pred_p.shape[0], 1])

        return [mu_p], [pred_p]

