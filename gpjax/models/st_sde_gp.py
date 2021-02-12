from . import Model, SDE_GP
from .. import Parameter
from .. import Kernel
from .. import Likelihood
from .. import Inference


from ..data import Data, SpatioTemporalData
from..distributions import KernelGaussianDistribution

from ..data.utils import *

from ..inference import StateSpace
from ..decorators import *
from ..sparsity import *

from ..settings import Settings
from ..computation.general import get_computational_primitives, kf_cholesky_solve_trace, cholesky_solve
from ..computation.kalman_filter import kalman_loop, kalman_loop_store_intermediate
from ..computation.rts_smoother import rts_smoother

from .. import Dispatcher

import jax.numpy as np
from jax import jit, partial

import numpy as onp

import typing 
from typing import Union, List, Optional

@Dispatcher.register('gp_model', StateSpace, None, None)
class ST_SDE_GP(SDE_GP):

    def setup(self):
        self.data = SpatioTemporalData(self.X, self.Y, self.X_onp, self.Y_onp)

        #if not self.set_defaults: return 

        self.prior_arr = []
        for p in range(self.num_outputs):
            prior = KernelGaussianDistribution(
                kernel=self.kernel_arr[p],
                meta={
                    'X': self.X
                }   
            ) 
            self.prior_arr.append(prior)

    @return_gradients
    @set_defaults_from_self
    def get_objective(self, data:Optional[Data] = None):
        latent = 0

        prior = self.prior_arr[latent]
        likelihood = self.likelihood
        Y = data.Y[latent]
        kernel = prior.kernel

        #select correct filter based on likelihood
        kf_fn = Dispatcher.dispatch('kalman_filter')

        neg_log_marg_lik, _, _, _, _, _ =  kf_fn(
            Y,
            data.N,
            data.dt,
            kernel,
            likelihood,
            self.sparsity,
            data.mask,
            store_intermediate=False
        )

        return neg_log_marg_lik    

    @set_defaults_from_self
    def posterior(self, data: Optional[Data]=None, diagonal_var: Optional[bool] = True):
        prior = self.prior_arr[0]
        likelihood = self.likelihood
        X = data.X[0]
        Y = data.Y[0]

        kernel = prior.kernel

        dt_all = data.dt
        Y_all = Y
        N_all = data.N
        mask = data.mask

        #TODO: diagonal_var=False returns the block diagonals, not the full
        mu, sig = self.filter_and_smooth(
            Y_all,
            N_all,
            dt_all,
            kernel,
            likelihood,
            self.sparsity,
            mask,
            store_intermediate=True
        )

        if diagonal_var:
            mu = np.reshape(mu, [mu.shape[0]*mu.shape[1], 1])

            sig = np.diagonal(sig, 0, 1, 2)
            sig = np.reshape(sig, [sig.shape[0]*sig.shape[1], 1])

        #TODO: for multiple latent support
        return [mu], [sig]

    @ensure_data_passed
    @set_defaults_from_self
    def predict_y(self, data_xs:Data, data: Optional[Data]=None, diagonal_var: Optional[bool] = True):

        if type(data_xs) == PlaceholderData:
            data_xs = SpatioTemporalData([data_xs.X], [data_xs.Y], [data_xs.X], [data_xs.Y])

        prior = self.prior_arr[0]
        likelihood = self.likelihood

        kernel = prior.kernel

        #organise data and prediction points into timeseries
        data_all = order_spatiotemporal_prediction(data_xs, data)

        #sort XS for prediction
        dt_all = data_all.dt
        Y_all = data_all.Y[0]
        N_all = data_all.N
        mask = data_all.mask

        mu, sig = self.filter_and_smooth(
            Y_all,
            N_all,
            dt_all,
            kernel,
            likelihood,
            self.sparsity,
            mask,
            store_intermediate=True
        )

        #extract predict locations from the predictions 
        mu, sig = unorder_spatiotemporal_prediction(data_all, mu, sig)

        if diagonal_var:
            sig = np.diagonal(sig, 0, 1, 2)

            mu = np.reshape(mu, [mu.shape[0]*mu.shape[1], 1])
            sig = np.reshape(sig, [sig.shape[0]*sig.shape[1], 1])

        return [mu], [sig]

    #@partial(jit, static_argnums=(2))
    def predict_f(self, XS:np.ndarray, diagonal_var:bool):
        raise NotImplementedError('Predict f is not implemented yet')

