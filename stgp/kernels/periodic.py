"""
See http://proceedings.mlr.press/v33/solin14.pdf
"""
import objax
import chex
import jax
import jax.numpy as np

from . import Kernel, MarkovKernel

from .. import Parameter

from tensorflow_probability.substrates import jax as tfp
from jax.scipy.linalg import expm


class _PeriodicBase(MarkovKernel):
    """ Two dimensional oscillatory SDE model """

    def __init__(self, frequency_param, lengthscale_param, variance_param, j):
        self.j = j

        self.frequency_param = frequency_param
        self.lengthscale_param = lengthscale_param
        self.variance_param = variance_param


    def to_ss(self, X_spatial=None):
        freq = self.frequency_param.value
        lengthscale = self.lengthscale_param.value
        variance= self.variance_param.value

        j = self.j

        F = np.array([
            [0, - freq * j],
            [freq * j, 0],
        ])

        L = np.eye(2)

        inv_ls = 1/(lengthscale**2)


        qj = 2 * (tfp.math.bessel_ive(j, inv_ls) / np.exp(-np.abs(inv_ls))) / np.exp(inv_ls)
        
        Qc = np.eye(2) * qj
        #Qc = np.array([
        #    [qj]
        #])

        H = np.array([
            [1.0, 0.0],
        ])


        Pinf = qj * np.eye(2)

        return F, L, Qc, H, Pinf

class Periodic(Kernel):
    def __init__(self, frequency, lengthscale, variance, active_dims = None):
        super(Periodic, self).__init__(input_dim=1, active_dims=active_dims)

        self.lengthscale_param = Parameter(lengthscale, constraint='positive', name='Periodic/lengthscale')
        self.variance_param = Parameter(variance, constraint='positive', name='Periodic/variance')
        self.frequency_param = Parameter(frequency, constraint='positive', name='Periodic/frequency')

    def K_diag(self, X1):
        variance = self.variance_param.value
        return variance * np.ones(X1.shape[0])

    def _K_scaler(self, x1, x2):
        """
        Computes: tbd
        """
        ls = self.lengthscale_param.value
        variance = self.variance_param.value
        frequency = self.frequency_param.value

        tau = np.abs(x1 - x2)

        k = variance * np.exp(
            - 2 * np.square(
                np.sin( frequency * tau / 2) / ls
            )
        )

        return k


class ApproxSDEPeriodic(MarkovKernel, Periodic):
    """ See TBD. """
    def __init__(self, period, lengthscale, variance, n_terms=10, active_dims = None):

        super(ApproxSDEPeriodic, self).__init__(period, lengthscale, variance, active_dims=active_dims)

        self.n_terms = n_terms

        self.base_kernel = None
        self._setup_base_kernel()

    def _setup_base_kernel(self):
        for n in range(self.n_terms):
            new_term = _PeriodicBase(
                self.period_param,
                self.lengthscale_param,
                self.variance_param,
                n+1
            )

            if self.base_kernel == None:
                self.base_kernel = new_term
            else:
                self.base_kernel = self.base_kernel + new_term

    def to_ss(self, X_spatial=None):
        return self.base_kernel.to_ss(X_spatial)


    def expm(self, dt, X_spatial=None):
        """appproximate matrix exponential A = expm(F * dt)"""

        F, _, _, _, _ = self.base_kernel.to_ss(X_spatial) 

        return expm( F * dt)





