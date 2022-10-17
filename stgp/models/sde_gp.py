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
from ..computation.filters import kalman_filter, rts_smoother

from ..defaults import get_default_likelihood
from ..data import TemporalData, SpatioTemporalData, get_sequential_data_obj
from ..data.sequential import add_temporal_points
from ..kernels import Matern32
from ..likelihood import get_product_likelihood
from ..transforms import Independent
from ..transforms.sdes import LTI_SDE

from ..defaults import get_default_kernel, get_default_likelihood, get_default_independent_prior
from ..sparsity import NoSparsity

@dispatch(Model, 'Sequential')
class SDE_GP(Posterior):
    def __new__(cls, data, *args, **kwargs):
        if isinstance(data, SpatioTemporalData):
            return ST_SDE_GP(data, *args, **kwargs)
        else:
            return T_SDE_GP(data, *args, **kwargs)

class BASE_SDE_GP(Posterior):
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
        super(BASE_SDE_GP, self).__init__(data=data)

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
            if self.kernel is None: 
                raise RuntimeError('Kernel must be passed!')

            if type(self.kernel) is not list:
                self.kernel = [self.kernel]

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
            self._prior = LTI_SDE(self._prior)

        if self.likelihood == None:
            self._likelihood = get_default_likelihood(self.output_dim)[0]

        if type(self.likelihood) == list:
            self._likelihood = get_product_likelihood(self._likelihood)


    def log_marginal_likelihood(self):
        lml, _  = kalman_filter.filter_loop(
            self.data,
            self.prior,
            self.likelihood.variance
        )

        return lml

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

    def filter_and_smooth(self, data, prior, R, full_state=False):
        _, kf_res  = kalman_filter.filter_loop(
            data,
            prior,
            R
        ) 

        return rts_smoother.smoother_loop(
            data, 
            prior,
            kf_res,
            full_state=full_state
        )

    def posterior_blocks(self):
        mu, var = self.filter_and_smooth(
            self.data,
            self.prior,
            self.likelihood.variance
        )
        mu = np.reshape(mu, [mu.shape[0], mu.shape[1]])
        return mu, var

    def posterior(self, diagonal=True, full_state=False):

        mu, var = self.filter_and_smooth(
            self.data,
            self.prior,
            self.get_likelihood_for_prediction(self.data),
            full_state = full_state
        )


        # only fix shapes is not returning the full-state
        if full_state is False:
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

        mu, var = self.filter_and_smooth(
            test_data,
            self.prior,
            self.get_likelihood_for_prediction(test_data)
        )

        # mu, var are in time - space format
        # Therefore we just need to stack them
        mu = np.reshape(mu, [-1, self.output_dim])
        var = np.reshape(var, [-1, self.output_dim, self.output_dim])

        # Unsort data and remove the training data
        mu = test_data.unsort(mu)[self.data.N:]
        var = test_data.unsort(var)[self.data.N:]

        return mu, var


    def predict_y(self, XS, squeeze=True):
        pred_mu, pred_var = self.predict_f(XS, squeeze=squeeze)

        if len(pred_var.shape) == 2:
            # unsqueeze variance to add on likelihood
            pred_var = pred_var[..., None]

        pred_y_mu, pred_y_var = evoke('predict_y_diagonal', self, self.likelihood)(
            XS,  self.likelihood, pred_mu, pred_var
        )

        if squeeze:
            pred_y_mu = np.squeeze(pred_y_mu)
            pred_y_var = np.squeeze(pred_y_var)

        return pred_y_mu, pred_y_var

class T_SDE_GP(BASE_SDE_GP):
    """ Temporal SDE GP """

    def get_likelihood_for_prediction(self, data):
        # Currently the likelihood is only defined on the training points. 
        # But due to the implementation we need to provide likelihood values everywhere
        # Jax will silently wraps around in this setting if less data is passed through

        R = self.likelihood.variance
        #R = self.likelihood.likelihood_arr[0].variance

        out_dim = R.shape[-1]

        points_added = data.Nt-R.shape[0] 
        # Full R. This wont be used at locations without data so we just ignore
        R_tmp = np.tile(np.eye(out_dim), [points_added, 1, 1])
        R = np.vstack([R, R_tmp])
        R = R[data.unique_idx][data.sort_idx]
        R = np.reshape(R, [data.Nt, self.likelihood.block_size, self.likelihood.block_size])

        return R

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


        mu, var = self.filter_and_smooth(
            test_data,
            self.prior,
            self.get_likelihood_for_prediction(test_data)
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

class ST_SDE_GP(BASE_SDE_GP):
    """ Spatio-Temporal SDE GP """

    def get_likelihood_for_prediction(self, data):
        # Currently the likelihood is only defined on the training points. 
        # But due to the implementation we need to provide likelihood values everywhere
        # Jax will silently wraps around in this setting if less data is passed through

        R = self.likelihood.variance

        # Data is temporal data. We do not use the Kalman filter and smoother to predict in space,
        #  only in time.  

        out_dim = self.likelihood.block_size

        points_added = data.Nt-R.shape[0] 
        # Full R. This wont be used at locations without data so we just ignore
        R_tmp = np.tile(np.eye(out_dim), [points_added, 1, 1])
        R = np.vstack([R, R_tmp])
        R = R[data.unique_idx][data.sort_idx]
        R = np.reshape(R, [data.Nt, self.likelihood.block_size, self.likelihood.block_size])

        return R

    def predict_f(self, XS: np.ndarray, diagonal=True, squeeze=False):
        """
        We use the Kalman filter and smoother to predict and the temporal slices of XS,
        and then use the results to extrapolate to the new spatial locations.

        Whilst we could use the Kalman smoother to predict at all these locations, having them 
        separate requires less pre-processing of the data, and having them separate is required for
        CVI anyway.

        In:
            XS: Ns x D

        """

        if diagonal is False:
            raise NotImplementedError()

        chex.assert_equal(XS.shape[1], self.data.D)

        NS = XS.shape[0]
        YS_nans = onp.NaN * onp.ones([NS, self.output_dim])

        XS_data = get_sequential_data_obj(
            XS, 
            YS_nans,
            sort=True
        )

        # Get all ordered temporal points
        # This will be uses to unsort the results
        all_t = np.vstack([self.data.X_time[:, None], XS_data.X_time[:, None]])
        all_temporal_data = get_sequential_data_obj(
            all_t,
            np.ones_like(all_t), # Dummy data, we only care about X here
            sort=True
        )

        X = onp.array(self.data.X)
        Y = onp.reshape(self.data.Y_flat, [-1, self.output_dim])

        XS_new = add_temporal_points(XS_data, self.data)
        YS_new_nans = onp.NaN * onp.ones([XS_new.shape[0], self.output_dim])

        # Stack X first so that training data does not get removed when sorting data
        X_stacked = onp.vstack([X, XS_new])
        Y_stacked = onp.vstack([Y, YS_new_nans])

        test_data = get_sequential_data_obj(
            X_stacked,
            Y_stacked,
            sort=True 
        )

        mu, var = self.filter_and_smooth(
            test_data,
            self.prior,
            self.get_likelihood_for_prediction(all_temporal_data)
        )

        mu, var_diag = evoke('spatial_conditional', XS_data, test_data, self, self.prior)(
            XS_data, test_data, mu, var, self, True
        )

        # mu/var is in latent-temporal-spatial format
        # Convert to temporal-spatial-latent format

        mu = np.transpose(mu, [1, 2, 0, 3])
        var_diag = np.transpose(var_diag, [1, 2, 0, 3])

        # Unsort data and remove the training data
        mu = all_temporal_data.unsort(mu)[self.data.Nt:]
        var_diag = all_temporal_data.unsort(var_diag)[self.data.Nt:]


        # mu, var are in time - space format
        # Therefore we just need to stack them
        mu = np.reshape(mu, [-1, self.output_dim])
        var_diag = np.reshape(var_diag, [-1, self.output_dim])

        return mu, var_diag
