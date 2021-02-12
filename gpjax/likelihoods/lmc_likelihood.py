from . import Likelihood
from ..parameter import Parameter
from ..computation import *

from ..decorators import return_gradients

import gpjax
import jax
import jax.numpy as np
from jax.experimental import loops
import numpy as onp

import typing
from typing import List, Optional

class LMC_Likelihood(Likelihood):
    def __init__(self, num_outputs:Optional[int] =1, num_latents:Optional[int]=1, variances:Optional[np.ndarray]=None, mixing_weights: Optional[np.ndarray]=None, trainable:Optional[bool]=True):
        self.name = 'LMC_Likelihood'

        self.num_outputs = num_outputs
        self.num_latents = num_latents

        super(LMC_Likelihood, self).__init__(name=self.name, meta={}, trainable=trainable)

        if variances is not None:
            if variances.shape[0] > 1 and variances.shape[0] != num_outputs:
                raise RuntimeError(self.name, ': Number of variances {var_num} should match number of outputs {num_outputs}'.format(var_num=variances.shape[0], num_outputs=num_outputs))
        else:
            variances = [1.0 for output in range(num_outputs)] if variances is None else variances
            variances = np.array(variances)

        self.variances = self.parameter(val=variances, constraint='positive', train=trainable, module_name=self.name, param_name='variance')

        if mixing_weights is None:
            mixing_weights = onp.random.randn(self.num_outputs, self.num_latents)

        self.mixing_weights = self.parameter(val=mixing_weights, train=trainable, module_name=self.name, param_name='mixing_weights')


    @property
    def coregion_weights(self):
        return self.mixing_weights.val

    def log_likelihood(self, Y, F_mu):
        return NotImplementedError()

    @property
    def variance(self):
        return self.variances.val

    def predict_mean_var(self, latent_mu_arr, latent_var_arr):
        mu_arr = []
        sig_arr = []
        mixing_weights = self.coregion_weights
        num_latents = self.num_latents
        variance = self.variance

        for output in range(self.num_outputs):
            output_weights = mixing_weights[output, :]
            mu = np.sum([output_weights[q]*latent_mu_arr[q] for q in range(num_latents)], axis=0)
            sig = np.sum([(output_weights[q]**2)*latent_var_arr[q] for q in range(num_latents)], axis=0)
            sig = sig+variance[output]
            sig = np.reshape(sig, [sig.shape[0], 1])

            mu_arr.append(mu)
            sig_arr.append(sig)

        return mu_arr, sig_arr



