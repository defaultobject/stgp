import objax
import jax
import jax.numpy as np
import numpy as onp
import chex
from typing import Optional, Tuple
import warnings

from ..obj_dispatch import obj_dispatch, obj_find
from ..dispatch import evoke
from ..core import Model, Posterior
from . import GP, BatchGP
from ..computation.filtering import sequential_kalman_filter, filter_and_smooth
from ..defaults import get_default_likelihood

from ..data.sequential import order_sequentially

@obj_dispatch(Model, 'Markov', 'NoSparsity')
class SDE_GP(Posterior):
    def __init__(
        self, 
        X=None, 
        Y=None, 
        Z = None,
        inference: 'Markov'=None, 
        likelihood: 'Likelihood'=None, 
        kernel: 'Kernel'=None, 
        prior: 'Transform'=None, 
        whiten=False, 
        fix_input=True,
        **kwargs
    ):
        self.raw_X = X
        self.raw_Y = Y
        self.raw_N = Y.shape[0]

        if fix_input:
            unique_idx, sort_idx, X_sorted, Y_sorted = order_sequentially(X, Y)
        else:
            unique_idx, sort_idx, X_sorted, Y_sorted = None, None, X, Y

        self.raw_N_time = Y_sorted.shape[0]

        # Use the sorted X and Y to construct the model on
        super(SDE_GP, self).__init__(X_sorted, Y_sorted)

        self.sort_idx = sort_idx
        self.unique_idx = unique_idx
        self._likelihood = likelihood
        self.kernel = kernel

        self.set_defaults()

    @property
    def likelihood(self): return self._likelihood 

    @property
    def input_space_dim(self): return self.X.shape[1]

    @property
    def output_dim(self): return 1

    def set_defaults(self):
        """ Replace missing options with defaults """

        # Only set a default kernel if we are in kernel mode and one has not been passed
        if self.kernel is None:
            warnings.warn('Using default Matern32 kernel with lengthscale 1.o')
            self.kernel = Matern32(lengthscales=[1.0])


        if self.likelihood == None:
            self._likelihood = get_default_likelihood(self.output_dim)[0]

    def log_marginal_likelihood(self, X: Optional[np.ndarray] = None, Y: Optional[np.ndarray] = None):
        return sequential_kalman_filter(
            self.X,
            self.Y,
            self.kernel,
            self.likelihood,
            N = self.raw_N_time
        )

    def get_objective(self, X=None, Y=None):
        return -self.log_marginal_likelihood(X, Y)

    def mean(self, XS):
        mu, _ = self.predict_f(XS, diagonal=True, squeeze=False)
        return mu

    def var(self, XS):
        _, var = self.predict_f(XS, diagonal=True, squeeze=False)
        return var

    def covar(self, XS_1, XS_2, X=None, Y=None):
        raise NotImplementedError()

    def predict_f(self, XS: np.ndarray, X: Optional[np.ndarray] = None, Y: Optional[np.ndarray] = None):
        NS = XS.shape[0]

        X = self.raw_X
        Y = self.raw_Y

        # Stack X first so that training data does not get removed when sorting data
        X_stacked = onp.vstack([X, XS])

        Y_nans = onp.NaN * onp.ones([NS, 1])
        Y_stacked = onp.vstack([Y, Y_nans])

        unique_idx, sort_idx, X_sorted, Y_sorted = order_sequentially(X_stacked, Y_stacked)

        N = X_sorted.shape[0]

        X_sorted = objax.StateVar(np.array(X_sorted))
        Y_sorted = objax.StateVar(np.array(Y_sorted))

        _, mu, var = filter_and_smooth(
            X_sorted.value,
            Y_sorted.value,
            self.kernel,
            self.likelihood,
            N = N
        )

        mu = mu.reshape([-1, 1])
        var = var.reshape([-1, 1])

        # unsort
        mu = mu[sort_idx][unique_idx][self.raw_N:]
        var = var[sort_idx][unique_idx][self.raw_N:]

        return mu, var

    def predict_y(self, XS):
        raise NotImplementedError()

