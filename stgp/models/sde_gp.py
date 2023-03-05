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
from ..computation.filters import kalman_filter, rts_smoother, parallel_kalman_filter, parallel_rts_smoother
from ..computation.matrix_ops import batched_block_diagional
from ..computation.permutations import permute_mat, permute_vec

from ..defaults import get_default_likelihood
from ..data import TemporalData, SpatioTemporalData, get_sequential_data_obj, SpatialTemporalInput
from ..data.sequential import add_temporal_points
from ..kernels import Matern32
from ..likelihood import get_product_likelihood, ProductLikelihood
from ..transforms import Independent
from ..transforms.sdes import LTI_SDE, LTI_SDE_Full_State_Obs



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
        full_state_observed = False,
        parallel = False,
        **kwargs
    ):
        # Use the sorted X and Y to construct the model on
        super(BASE_SDE_GP, self).__init__(data=data)

        self._likelihood = likelihood
        self.kernel = kernel
        self._prior = prior

        self.full_state_observed = full_state_observed

        self.set_defaults()

        self.parallel  = parallel

        

    @property
    def likelihood(self): return self._likelihood 

    @property
    def input_space_dim(self): return self.data.X.shape[1]

    @property
    def output_dim(self): 
        P = self.data.P

        if self.full_state_observed:

            out_dim = self.prior.spatial_output_dim*self.prior.temporal_output_dim
            return out_dim

            if self.kernel is not None:
                return self.kernel.state_space_dim() 
            else:
                return self.prior.output_dim 

        return P

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
            if self.full_state_observed:
                self._prior = LTI_SDE_Full_State_Obs(self._prior)
            else:
                self._prior = LTI_SDE(self._prior)

        if self.likelihood == None:
            self._likelihood = get_default_likelihood(self.output_dim)[0]

        if type(self.likelihood) == list:
            self._likelihood = get_product_likelihood(self._likelihood)


    def log_marginal_likelihood(self):
        lml, _  = kalman_filter.filter_loop(
            self.data,
            self.prior,
            self.likelihood.variance,
            parallel = self.parallel
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
            R,
            parallel = self.parallel
        ) 

        return rts_smoother.smoother_loop(
            data, 
            prior,
            kf_res,
            full_state=full_state,
            parallel = self.parallel
        )

    def posterior_blocks(self):
        """ Compute the posterior p(f_t | Y) for all t in time-latent-space format.  """
        mu, var = self.filter_and_smooth(
            self.data,
            self.prior,
            self.likelihood.variance
        )

        var = var[:, None, ...]

        # in time-latent-space format
        chex.assert_rank([mu, var], [3, 4])
        return mu, var

    def posterior(self, diagonal=True, full_state=False):
        mu, var = self.filter_and_smooth(
            self.data,
            self.prior,
            #self.get_likelihood_for_prediction(self.data),
            self.likelihood.variance,
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

        # mu, var are in time - latent- space format but space is 1
        # Therefore we just need to stack them
        mu = np.reshape(mu, [-1, self.output_dim])

        # only keep diagonals
        if diagonal:
            var_diag = np.diagonal(var, axis1=1, axis2=2)
            var_diag = np.reshape(var_diag, [-1, self.output_dim])
            var = test_data.unsort(var_diag)[self.data.N:]
        else:
            var = test_data.unsort(var)[self.data.N:]

        # Unsort data and remove the training data
        mu = test_data.unsort(mu)[self.data.N:]

        if squeeze:
            mu = np.squeeze(mu)
            var = np.squeeze(var)
        else:
            # ensure rank 3 and 4
            mu = mu[..., None]
            var = var[:, None, ...]

            if diagonal:
                # add missing diagonal axid which is removed when extracting the diagonal
                var = var[..., None]

            chex.assert_rank([mu, var], [3, 4])

        return mu, var

class ST_SDE_GP(BASE_SDE_GP):
    """ Spatio-Temporal SDE GP """

    def get_likelihood_for_prediction(self, data):
        # Currently the likelihood is only defined on the training points. 
        # But due to the implementation we need to provide likelihood values everywhere
        # Jax will silently wraps around in this setting if less data is passed through

        R = self.likelihood.variance

        # Data is temporal data. We do not use the Kalman filter and smoother to predict in space,
        #  only in time.  

        if isinstance(self.likelihood, ProductLikelihood) or issubclass(type(self.likelihood), ProductLikelihood):
            assert len(self.likelihood.likelihood_arr) == 1
            out_dim = self.likelihood.likelihood_arr[0].block_size

        else:
            out_dim = self.likelihood.block_size


        points_added = data.Nt-R.shape[0] 
        # Full R. This wont be used at locations without data so we just ignore
        R_tmp = np.tile(np.eye(out_dim), [points_added, 1, 1])
        R = np.vstack([R, R_tmp])
        R = R[data.unique_idx][data.sort_idx]
        R = np.reshape(R, [data.Nt, out_dim, out_dim])

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


        When diagonal is True we return
            mu;
            var:

        When diagonal is False we return the block diagonal across latents
        """
        chex.assert_equal(XS.shape[1], self.data.D)

        # Convert XS to a spatio-temporal object
        NS = XS.shape[0]
        # in data-latent format
        YS_nans = onp.NaN * onp.ones([NS, self.output_dim]) # dummy Y values

        # Convert XS to time-space format
        XS_data = get_sequential_data_obj(
            XS, 
            YS_nans,
            sort=True
        )
        # The KF is used to predict at new time points. Collect the order temporal points across
        #    trainig and testing data.
        # This will also be used to unsort the results
        # NOTE: self.data must go before XS_data otherwise data points can be overwritten by the sorting
        #    as only unique points are kept
        all_t = np.vstack([self.data.X_time[:, None], XS_data.X_time[:, None]])
        all_temporal_data = get_sequential_data_obj(
            all_t,
            np.ones_like(all_t), # Dummy data, we only care about X here
            sort=True
        )

        dummy_training_data = get_sequential_data_obj(
            SpatialTemporalInput(
                self.data.X_time, 
                np.tile(np.arange(self.data.X_space.shape[0])[:, None], [1, self.data.X_space.shape[1]])
            ),
            self.data.Y_st,
            sort=False 
        )

        # Collect Training Data
        X = onp.array(dummy_training_data.X)

        # self.data.Y is stored in time-space-latent format, reshape into data-latent
        Y = onp.reshape(dummy_training_data.Y_flat, [-1, self.output_dim])

        # create new data with the same spatial points as self.data but with all time points across XS and X
        XS_temporal_new = add_temporal_points(XS_data, dummy_training_data)
        YS_temporal_new_nans = onp.NaN * onp.ones([XS_temporal_new.shape[0], self.output_dim]) # data-latent format

        # Stack X first so that training data does not get removed when sorting data
        X_stacked = onp.vstack([X, XS_temporal_new])
        Y_stacked = onp.vstack([Y, YS_temporal_new_nans])

        # ST data object across all (unique) training and testing temporal points but only 
        #   at the training spatial locations

        # this should not sort space!!
        # we should not be sorting space for test_data as this is also used for induicng points
        # . where ordering in space is not guarenteed
        # so we first order to get the unique points
        temporal_test_data = get_sequential_data_obj(
            X_stacked,
            Y_stacked,
            sort=True 
        )

        XS_temporal_new = add_temporal_points(XS_data, self.data)
        YS_temporal_new_nans = onp.NaN * onp.ones([XS_temporal_new.shape[0], self.output_dim]) # data-latent format

        # Stack X first so that training data does not get removed when sorting data
        Y_stacked = onp.vstack([self.data.Y, YS_temporal_new_nans])

        _X = X_stacked[temporal_test_data.unique_idx][temporal_test_data.sort_idx]
        _Y = Y_stacked[temporal_test_data.unique_idx][temporal_test_data.sort_idx]


        # we now have a spatio-temporal grid where the spatial part is unchanged
        temporal_test_data = get_sequential_data_obj(
            SpatialTemporalInput(
                temporal_test_data.X_time, 
                self.data.X_space,
            ),
            np.transpose(np.reshape(_Y, [-1, self.data.Ns, self.data.P]), [0, 2, 1]),
            sort=False
        )


        # Compute posterior at temporal_test_data
        mu_t, var_t = self.filter_and_smooth(
            temporal_test_data,
            self.prior,
            self.get_likelihood_for_prediction(all_temporal_data)
        )
        mu_p = jax.vmap(lambda a: permute_vec(a, 2))(mu_t)
        mu_p = np.reshape(mu_p, [-1, 2, 1])


        # construct testing data at new spatial locations
        XS_spatial_new = add_temporal_points(all_temporal_data, XS_data)
        YS_spatial_new_nans = onp.NaN * onp.ones([XS_spatial_new.shape[0], self.output_dim])

        xs_spatial_data = get_sequential_data_obj(
            XS_spatial_new,
            YS_spatial_new_nans,
            sort=True 
        )

        # mu_t and var_t are in time-latent-space format
        # when predicting we only predict f, not the state as well
        if not self.full_state_observed:
            # remove the extra state dims
            mu_t = mu_t[:, :self.data.Ns, :]
            var_t = var_t[:, :self.data.Ns, :][:, :, :self.data.Ns]

        if True:
            # Compute spatial conditions to get posterior at new spatial points
            mu, var = evoke('spatial_conditional', XS_data, temporal_test_data, self, self.prior)(
                xs_spatial_data, temporal_test_data, mu_t, var_t, self, False
            )
        else:
            mu, var = mu_t, var_t[:, None, ...]

        # mu/var is in  time - (latent x space) format
        # Unsort data and remove the training data
        mu_time_unsorted = all_temporal_data.unsort(mu)[self.data.Nt:]
        var_time_unsorted = all_temporal_data.unsort(var)[self.data.Nt:]


        # convert to time-space-latent format
        if self.full_state_observed:
            out_dim = self.prior.spatial_output_dim*self.prior.temporal_output_dim

            mu_p = jax.vmap(lambda a: permute_vec(a, out_dim))(mu_time_unsorted)
            var_p = jax.vmap(lambda A: permute_mat(A[0], out_dim))(var_time_unsorted)

            mu_p = np.reshape(mu_p, [-1, out_dim, 1])
            var_p = batched_block_diagional(var_p, out_dim)
            var_p = np.reshape(var_p, [-1, 1, out_dim, out_dim])
        else:
            mu_p = mu_time_unsorted
            var_p = var_time_unsorted

            # time - space format
            mu_p = np.reshape(mu_p, [-1, 1, 1])
            var_p = np.reshape(
                np.diagonal(var_p, axis1=2, axis2=3),
                [-1, 1, 1, 1]
            )
        
        # unsort to original permutation in XS
        mu_p_unsorted = XS_data.unsort(mu_p)
        var_p_unsorted = XS_data.unsort(var_p)



        return mu_p_unsorted, var_p_unsorted
