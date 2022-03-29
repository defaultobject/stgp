import objax
import jax
import jax.numpy as np
import numpy as onp
import chex
from typing import Optional, Tuple
import warnings

from ..dispatch import dispatch
from ..dispatch import evoke
from ..core import Model, Posterior
from . import GP, BatchGP
from ..computation.filtering import sequential_kalman_filter, filter_and_smooth
from ..defaults import get_default_likelihood
from ..data import TemporalData, SpatioTemporalData, get_sequential_data_obj
from ..kernels import Matern32
from ..likelihood import get_product_likelihood
from ..transforms import Independent

from ..defaults import get_default_kernel, get_default_likelihood, get_default_independent_prior
from ..sparsity import NoSparsity

@dispatch(Model, 'Sequential')
class SDE_GP(Posterior):
    def __init__(
        self, 
        data = None,
        Z = None,
        inference: 'Sequential'=None, 
        likelihood: 'Likelihood'=None, 
        kernel: 'Kernel'=None, 
        prior: 'Transform'=None, 
        whiten=False, 
        fix_input=True,
        **kwargs
    ):
        # Use the sorted X and Y to construct the model on
        super(SDE_GP, self).__init__(data=data)

        self._likelihood = likelihood
        self.kernel = kernel
        self._prior = prior

        self.set_defaults()

    @property
    def likelihood(self): return self._likelihood 

    @property
    def input_space_dim(self): return self.data.X.shape[1]

    @property
    def output_dim(self): return self.data.Y.shape[-1]

    @property
    def X(self): return self.data.X 

    @property
    def Y(self): return self.data.Y 

    @property
    def prior(self): return self._prior


    @property
    def Nt(self): 
        """ Return the number of temporal points. """
        return self.data.Nt 

    def set_defaults(self):
        """ Replace missing options with defaults """

        if (self.kernel is not None) and (self.prior is not None):
            raise RuntimeError('Only kernel or a prior must be passed')

        if self.prior is None:

            # Only set a default kernel if we are in kernel mode and one has not been passed
            warnings.warn('Using default Matern32 kernel with lengthscale 1.0')
            self.kernel = [Matern32(lengthscales=[1.0]) for q in range(self.output_dim)]


            # Pass object to avoid storing multiple copies of X
            X_ref = self.data._X
            sparsity = [
                NoSparsity(Z_ref = X_ref) 
                for q in range(self.output_dim)
            ]

            # Construct independent prior
            self._prior = get_default_independent_prior(
                sparsity,
                self.input_space_dim, 
                self.output_dim, 
                kernel_list=self.kernel
            )

        if type(self.prior) != Independent:
            self._prior = Independent([self._prior])

        if self.likelihood == None:
            self._likelihood = get_default_likelihood(self.output_dim)[0]

        if type(self.likelihood) == list:
            self._likelihood = get_product_likelihood(self._likelihood)


    def log_marginal_likelihood(self):
        return sequential_kalman_filter(
            self.data,
            self.prior,
            self.likelihood,
            N = self.Nt
        )

    def get_objective(self):
        return -self.log_marginal_likelihood()

    def mean(self, XS):
        mu, _ = self.predict_f(XS, diagonal=True, squeeze=False)
        return mu

    def var(self, XS):
        _, var = self.predict_f(XS, diagonal=True, squeeze=False)
        return var

    def covar(self, XS_1, XS_2, X=None, Y=None):
        raise NotImplementedError()

    def jittable_predict_f(self, XS, YS, nan_grid_X, nan_grid_Y, sort_idx, return_idx):
        """
        Due to the sorting required to into a spatio-temporal grid we require a separate prediction function that passes through the indexes required to sort.
        """
        raise NotImplementedError()
        X = self.raw_X
        Y = self.raw_Y
        X_stacked = np.vstack([X, XS, nan_grid_X])
        Y_stacked = np.vstack([Y, YS, nan_grid_Y])

        X_sorted = X_stacked[sort_idx]
        Y_sorted = Y_stacked[sort_idx]

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
        mu = mu[return_idx]
        var = var[return_idx]

        return mu, var

    def posterior_blocks(self):
        _, mu, var = filter_and_smooth(
            self.data,
            self.prior,
            self.likelihood,
            N = self.data.Nt
        )
        return mu, var

    def posterior(self, diagonal=True):
        _, mu, var = filter_and_smooth(
            self.data,
            self.prior,
            self.likelihood,
            N = self.data.Nt
        )


        # mu, var are in time - space format
        # Therefore we just need to stack them
        mu = np.reshape(mu, [-1, 1])

        # only keep diagonals
        if diagonal:
            var_diag = np.diagonal(var, axis1=1, axis2=2)
            var_diag = np.reshape(var_diag, [-1, 1])

            return mu, var_diag

        return mu, var

    def predict_blocks(self, XS, group_size, block_size, diagonal=False):
        chex.assert_equal(group_size, 1)

        NS = XS.shape[0]
        chex.assert_equal(XS.shape[1], self.data.D)

        X = onp.array(self.data.X)
        Y = onp.reshape(self.data.Y, [-1, self.output_dim])

        # Stack X first so that training data does not get removed when sorting data
        X_stacked = onp.vstack([X, XS])

        Y_nans = onp.NaN * onp.ones([NS, self.output_dim])
        Y_stacked = onp.vstack([Y, Y_nans])

        test_data = get_sequential_data_obj(
            X_stacked,
            Y_stacked,
            sort=True 
        )

        _, mu, var = filter_and_smooth(
            test_data,
            self.prior,
            self.likelihood,
            N = test_data.Nt
        )

        # mu, var are in time - space format
        # Therefore we just need to stack them
        mu = np.reshape(mu, [-1, self.output_dim])
        var = np.reshape(var, [-1, self.output_dim, self.output_dim])

        # Unsort data and remove the training data
        mu = test_data.unsort(mu)[self.data.N:]
        var = test_data.unsort(var)[self.data.N:]

        return mu, var


    def predict_f(self, XS: np.ndarray, diagonal=True, squeeze=False):

        if diagonal is False:
            raise NotImplementedError()

        NS = XS.shape[0]
        chex.assert_equal(XS.shape[1], self.data.D)

        X = onp.array(self.data.X)
        Y = onp.reshape(self.data.Y_flat, [-1, self.output_dim])

        # Stack X first so that training data does not get removed when sorting data
        X_stacked = onp.vstack([X, XS])

        Y_nans = onp.NaN * onp.ones([NS, self.output_dim])
        Y_stacked = onp.vstack([Y, Y_nans])


        test_data = get_sequential_data_obj(
            X_stacked,
            Y_stacked,
            sort=True 
        )


        _, mu, var = filter_and_smooth(
            test_data,
            self.prior,
            self.likelihood,
            N = test_data.Nt
        )

        # mu, var are in time - space format
        # Therefore we just need to stack them
        mu = np.reshape(mu, [-1, self.output_dim])

        # only keep diagonals
        var_diag = np.diagonal(var, axis1=1, axis2=2)
        var_diag = np.reshape(var_diag, [-1, self.output_dim])

        # Unsort data and remove the training data
        mu = test_data.unsort(mu)[self.data.N:]
        var_diag = test_data.unsort(var_diag)[self.data.N:]

        return mu, var_diag

    def predict_y(self, XS, squeeze=True):
        pred_mu, pred_var = self.predict_f(XS, squeeze=squeeze)

        # TODO: fix the hack
        pred_y_mu, pred_y_var = evoke('predict_y_diagonal', 'BatchGP', self.likelihood)(
            XS,  self.likelihood, pred_mu, pred_var
        )

        return pred_y_mu, pred_y_var


