from . import Model
from .. import Parameter
from .. import Kernel
from .. import Likelihood
from .. import Sparsity

from ..data import Data, ListData

from ..distributions import *
from ..decorators import *

from ..settings import Settings

from ..approximate_posteriors import GaussianApproxPosterior, MeanFieldApproxPosterior
from ..sparsity import NoSparsity
from ..inference import VariationalInference

from .. import Dispatcher

import jax.numpy as np
from jax import jit, partial

import numpy as onp

import typing 
from typing import Union, List, Optional

import warnings



@Dispatcher.register('gp_model', VariationalInference, None, None)
class VGP(Model):
    def setup(self):
        """
            It is assumed that if the observations are passed as a list of input output arrays (e.g [X for i in num_latents])
                then this will create num_latent number of independent VGP objectives. The final objective is then just the sum of the 
                individual objectives.
        """

        self.data = ListData(self.X, self.Y, self.X_onp, self.Y_onp)

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
            q = MeanFieldApproxPosterior()
            self.inference.initialize(q)

        self.inference.variational_posterior.setup(self.N_latents)

        #=============================setup prior=============================
        self.prior_arr = []
        for p in range(self.num_latents):

            if type(self.sparsity) is not NoSparsity:
                X = self.sparsity.Z[p]
            else:
                X = self.X_arr[p]

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

    @return_gradients
    @set_defaults_from_self
    def get_objective(self, data:Optional[Data] = None):
        elbo = 0.0

        X_arr = data.X

        debug = False

        for latent in range(self.num_latents):

            if type(self.sparsity) is not NoSparsity:
                Z = self.sparsity_arr[latent].Z
            else:
                Z = X_arr[latent]

            q_p = self.inference.variational_posterior.components[latent]

            ell = q_p.expected_log_likelihood(data, self.likelihood, self, latent)

            kl = q_p.KL(Z, self.prior_arr[latent])

            if debug:
                print('ell: ', ell)
                print('kl: ', kl)

            elbo_l =  ell-kl 

            elbo += elbo_l

        return -np.squeeze(elbo)

        
    @ensure_data_passed
    @set_defaults_from_self
    def predict_y(self, data_xs:Data, data:Optional[Data] = None, diagonal_var: Optional[bool]=True):
        mu_arr = []
        pred_arr = []

        for latent in range(self.num_latents):
            q_p = self.inference.variational_posterior.components[latent]
            mu_p, pred_p = q_p.predict_y(data_xs, data, self, latent, diagonal_var)



            if diagonal_var:
                #ensure correct shape
                if type(mu_p) is list:
                    mu_p = mu_p[0]
                    pred_p = pred_p[0]

                mu_p = np.squeeze(mu_p).reshape([-1, 1])
                pred_p = np.squeeze(pred_p).reshape([-1, 1])

            mu_arr.append(mu_p)
            pred_arr.append(pred_p)
        
        return mu_arr, pred_arr


    @set_defaults_from_self
    def posterior(self, data: Optional[Data]=None, diagonal_var: Optional[bool] = True):
        raise NotImplementedError()

    @ensure_data_passed
    @set_defaults_from_self
    def predict_f(self, data_xs:Data, data:Optional[Data] = None, diagonal_var: Optional[bool]=True):
        mu_arr = []
        pred_arr = []

        for latent in range(self.num_latents):
            q_p = self.inference.variational_posterior.components[latent]
            mu_p, pred_p = q_p.predict_latent(data_xs, data, self, latent, diagonal_var)



            if diagonal_var:
                #ensure correct shape
                if type(mu_p) is list:
                    mu_p = mu_p[0]
                    pred_p = pred_p[0]

                mu_p = np.squeeze(mu_p).reshape([-1, 1])
                pred_p = np.squeeze(pred_p).reshape([-1, 1])

            mu_arr.append(mu_p)
            pred_arr.append(pred_p)
        
        return mu_arr, pred_arr
