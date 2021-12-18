import objax
import jax
import jax.numpy as np
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
            sort_idx, X_sorted, Y_sorted = order_sequentially(X, Y)
        else:
            sort_idx, X_sorted, Y_sorted = None, X, Y

        # Use the sorted X and Y to construct the model on
        super(SDE_GP, self).__init__(X_sorted, Y_sorted)

        self.sort_idx = sort_idx
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
            N = self.raw_N
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
        _, mu, var = filter_and_smooth(
            self.X,
            self.Y,
            self.kernel,
            self.likelihood,
            N = self.raw_N
        )

        return mu, var

    def predict_y(self, XS):
        raise NotImplementedError()

