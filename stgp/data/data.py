import objax
import chex
import jax
import jax.numpy as np
import numpy as onp
import objax

from .sequential import order_sequentially_np, pad_with_nan_to_make_grid
from .. import Parameter
from batchjax import batch_or_loop, BatchType
from ..utils.utils import get_batch_type
from ..utils.nan_utils import get_same_shape_mask

class Input(objax.Module):
    """ Base class for storing Input X.  """
    def __init__(self, X, name='X', train=False):
        """
        X can either be an array for another Input object. If X is an array we store it as a parameter, otherwise
            we just store the object.
        """

        if X is None:
            self._X = None
            self._X_ref = None
        elif isinstance(X, Input):
            self._X_ref = X # Store reference
            self._X = None
        else:
            self._X = Parameter(np.array(X), train=train, name=name)
            self._X_ref = None

    @property
    def shape(self):
        """ 
        Mimics np.ndarray to make it easier to pass around Input object and np.ndarrays interchangably
        """
        return self.X.shape

    @property
    def X(self):
        if self._X is not None:
            return self._X.value

        elif self._X_ref is not None:
            return self._X_ref.X

        return None

    @property
    def value(self):
        """ Mimic Parameter object. """
        return self.X


class SpatialTemporalInput(Input):
    def __init__(self, X_time, X_space, train=False):
        self._X_time = Input(X_time, name='X_Time', train=False)
        self._X_space = Input(X_space, name='X_Space', train=train)

        self.Nt = self._X_time.shape[0]
        self.Ns = self._X_space.shape[0]
        self.D = self._X_space.shape[1] + 1 # Plus one for the temporal dim

    @property
    def X_time(self):
        return self._X_time.X

    @property
    def X_space(self):
        return self._X_space.X

    @property
    def X(self):
        # We return X so that it is already ordered.
        #  ie in time-space ordering
        X_t = np.repeat(self.X_time[:, None], self.Ns)[:, None]
        X_s = np.tile(self.X_space, [self.Nt, 1])

        X = np.hstack([X_t, X_s])

        return X


class Data(objax.Module):
    def __init__(self, X, Y, minibatch_size=None):

        self._Y = Parameter(np.array(Y), train=False, name='Y')
        self.save_X(X, train=False, name='X')

        self.N = Y.shape[0]

        self.generator = objax.random.Generator(seed=0)

        if minibatch_size is not None:
            self.minibatch_size = minibatch_size
            self.minibatch = True
            self.idx = None
        else:
            self.minibatch_size = self.N
            self.minibatch = False
            self.idx = None

    def batch(self):
        self.idx = objax.random.randint(
            (self.minibatch_size,), 
            low=0, 
            high=self.N-1, 
            generator=self.generator
        )

    @property
    def base(self):
        return self

    @property
    def Y(self):
        if self.minibatch:
            return self._Y.value[self.idx]
        else:
            return self._Y.value

    @property
    def X(self):
        if self.minibatch:
            return self._X.X[self.idx]
        else:
            return self._X.X

    def save_X(self, _X, name='X', train=False):
        """
        To help reduce memory consumption we support passing X as a reference or an numpy/jax array.
        This provides a helper function to check which form is passed and saves it appropiately.
        """

        # Store data as objax parameters so that they can be used on GPUs etc
        if isinstance(_X, Input):
            self._X = _X # Store as reference
        else:
            self._X = Input(np.array(_X), name=name, train=train)

class TransformedData(Data):
    def __init__(self, base_data, transform_arr):
        self.transform_arr = objax.ModuleList(transform_arr)
        self.base_data = base_data

        self.minibatch = self.base_data.minibatch

        self.N = self.base_data.N

    def batch(self):
        return self.base_data.batch()

    def forward_transform(self, Y_base):
        P = Y_base.shape[1]
        # Compute lml for each likelihood and prior
        Y_transformed = batch_or_loop(
            lambda t_fn, y_p: t_fn.forward(y_p),
            [ self.transform_arr, Y_base ],
            [ 0, 1],
            dim = P,
            out_dim = 1,
            batch_type = get_batch_type(self.transform_arr)
        )

        # batch_or_loop assumes that the batchout output is axis 0
        # We want the same shape as Y_base so we transpose

        Y_transformed = Y_transformed.T

        chex.assert_equal_shape([Y_base, Y_transformed])

        return Y_transformed

    def inverse_transform(self, Y_base):
        P = Y_base.shape[1]
        # Compute lml for each likelihood and prior
        Y_transformed = batch_or_loop(
            lambda t_fn, y_p: t_fn.inverse(y_p),
            [ self.transform_arr, Y_base ],
            [ 0, 1],
            dim = P,
            out_dim = 1,
            batch_type = get_batch_type(self.transform_arr)
        )

        # batch_or_loop assumes that the batchout output is axis 0
        # We want the same shape as Y_base so we transpose

        Y_transformed = Y_transformed.T

        chex.assert_equal_shape([Y_base, Y_transformed])

        return Y_transformed

    @property
    def Y(self):
        Y_base = self.Y_base

        P = Y_base.shape[1]

        Y_base_mask =  get_same_shape_mask(Y_base)
        Y_base_no_nan = np.nan_to_num(Y_base, nan=0.0)

        # remove nans for the transform
        Y_transformed = self.forward_transform(Y_base_no_nan)

        # add back nans
        Y_transformed = Y_transformed + 0 * Y_base

        return Y_transformed

    def log_jacobian(self, Y_base):
        """ Computes log |dT(Y)/dY| """

        # Compute jacobian for each ouput
        P = Y_base.shape[1]

        # Compute lml for each likelihood and prior
        jac = batch_or_loop(
            lambda t_fn, y_p: jax.vmap(jax.grad(t_fn.forward))(y_p),
            [ self.transform_arr, Y_base ],
            [ 0, 1],
            dim = P,
            out_dim = 1,
            batch_type = get_batch_type(self.transform_arr)
        )

        jac = np.log(jac)

        jac = jac.T
        chex.assert_equal_shape([Y_base, jac])

        return jac

    @property
    def Y_base(self):
        return self.base_data.Y

    @property
    def X(self):
        return self.base_data.X

    @property
    def _X(self):
        return self.base_data._X


class AggregatedData(Data):
    pass

class SpatialAggregatedData(AggregatedData):
    pass

class TemporalAggregatedData(AggregatedData):
    pass

class SequentialData(Data):
    def __init__(self):

        self.minibatch = False


        self.unique_idx = None
        self.sort_idx = None
        self.points_added = None
        self.original_shape = None

    def sort(self, X, Y):
        """ 
        Converts (X, Y) into a data format that supports running Kalman filtering and smoothing algorithms. This is done by:
            1) First the data must lie on a (spatio-temporal) grid. This is done by padding the data with necessary missing/fake/nan observations.
            2) Second the data is sorted to ensure time-space format.
        """
        num_original_points = X.shape[0]


        # Adding missing points to make the full spatio-temporal grid
        points_added, X_padded, Y_padded = pad_with_nan_to_make_grid(
            X,
            Y
        )

        # Convert to time-space format
        unique_idx, reverse_unique_idx, sort_idx, X_sorted, Y_sorted = order_sequentially_np(
            X_padded, Y_padded
        )

        # Save sorting indexes so that this sorting function can be reversed
        self.num_original_points = num_original_points
        self.num_points_added = points_added
        self.unique_idx = unique_idx
        self.reverse_unique_idx = reverse_unique_idx
        self.sort_idx = sort_idx

        return X_sorted, Y_sorted

    @property
    def X_space(self):
        raise NotImplementedError()

    @property
    def X_time(self):
        raise NotImplementedError()

    @property
    def Y_st(self):
        raise NotImplementedError()

    @property
    def Y(self):
        raise NotImplementedError()

    @property
    def Y_flat(self):
        raise NotImplementedError()

    def unsort(self, A):
        """ Reverse the sorting steps performed in self.sort """
        return A[self.sort_idx][self.reverse_unique_idx][:self.num_original_points]

    @property
    def Y_flat(self):
        raise NotImplementedError()


class SpatioTemporalData(SequentialData):
    def __init__(self, X_time = None, X_space = None, X = None,  Y = None, sort=True):
        """
        Base class for Spatio-temporal Data


        There are two cases supported:

        1) X is already sorted and X_time + X_space are passed

            X_time: Nt  
            X_space: Ns x D
            X: None
            Y: Nt x Ns x P

        3) X is already sorted and X is passed

            X_time: None  
            X_space: None
            X: Nt * Ns x D
            Y: Nt x Ns x P

        2) X is not sorted 

            X_time: None
            X_space: None
            X: N x 1 
            Y: N x P

        If Y is None, then only X will be sorted

        """

        if sort:
            # X_time and X_space are None
            if X is None: raise RuntimeError('X must be passed')

            chex.assert_rank(X,  2)
            if Y is not None:
                chex.assert_rank(Y,  2)

            X_sorted, Y_sorted = self.sort(X, Y)

            # self.sort returns the full spatio-temporal dataset but we only require the temporal
            #  and spatial parts
            X_time = X_sorted[:, 0, 0]
            X_space = X_sorted[0, :, 1:]
            Y = Y_sorted


        if sort is True:
            chex.assert_rank(X_time, 1)
            chex.assert_rank(X_space, 2)
            self._X = SpatialTemporalInput(X_time, X_space, train=False)
        else:
            self._X = X

        if Y is not None:
            self._Y = Parameter(np.array(Y), train=False, name='Y')
            self.P = Y.shape[2]

        # Useful statistcs of the data
        self.Nt = self._X.Nt
        self.Ns = self._X.Ns
        self.D = self._X.D
        self.N = self.Ns*self.Nt

        self.check_shapes()

    def check_shapes(self):
        pass
        #chex.assert_rank(Y, 3)
        #chex.assert_equal(self._X.X_time.shape[0], Y.shape[0])
        #chex.assert_equal(self._X.X_space.shape[0], Y.shape[1])


    @property
    def X(self):
        """ Constructs the full spatio-temporal input from the temporal and spatial parts. """
        return self._X.X

    @property
    def X_space(self):
        return self._X.X_space

    @property
    def X_time(self):
        return self._X.X_time

    @property
    def Y_st(self):
        return self._Y.value

    @property
    def Y(self):
        return self.Y_flat

    @property
    def Y_flat(self):
        # X is returned in time-space ordering  so we return Y in the same order
        # Y is already sorted by time, and then by space.
        # Therefore all we have to is reshape

        Y = np.reshape(
            self.Y_st,
            [-1, self.P]
        )

        return Y

class DataReshape(SpatioTemporalData):
    """ Same functionality as SpatioTemporalData except Y is flat and is reshaped lazily """

    def __init__(self, data, new_shape):
        self.data = data
        self.new_shape = new_shape

    @property
    def base(self):
        return self.data.base

    @property
    def Y(self):
        Y_raw = self.Y_flat
        return np.reshape(Y_raw, self.new_shape)

    @property
    def Y_flat(self):
        return self.data._Y.value
    @property
    def Y_flat(self):
        return self.data._Y.value

    def __getattr__(self, name, *args, **kwargs):
        return getattr(self.data, name)

class TemporalData(SequentialData):
    def __init__(self, X, Y, sort=True, train_y = False):
        """
        There are two cases supported:

        1) X is already sorted 

            X: Nt x 1 
            Y: Nt x Ns x 1 = Nt x 1 x 1

        2) X is not sorted 

            X: Nt x 1 
            Y: Nt x 1 
        """

        super(TemporalData, self).__init__()

        # Only supports single output 
        X_time = X
        chex.assert_rank(X_time, 2)

        if sort:
            chex.assert_rank(Y, 2)
            chex.assert_equal(Y.shape[1], 1)

            # Sort
            X_sorted, Y_sorted = self.sort(
                X_time, Y
            )

            # self.sort sorts onto a spatio-temporal grid. We only require the temporal part.
            X_sorted = X_sorted[..., 0]

        else:
            X_sorted = X_time
            Y_sorted = Y   

        chex.assert_rank(X_sorted, 2)
        chex.assert_rank(Y_sorted, 3)

        self._Y = Parameter(np.array(Y_sorted), train=train_y, name='Y')
        self.save_X(X_sorted, train=False)

        # Useful statistcs of the data
        self.Nt = X_sorted.shape[0]
        self.Ns = 1
        self.D = 1
        self.N = self.Ns*self.Nt
        self.output_dim = 1
        self.P = 1

    @property
    def X_time(self):
        return self._X.X[:, 0]

    @property
    def X_space(self):
        return None

    @property
    def X(self):
        return self._X.X

    @property
    def Y_st(self):
        return self._Y.value

    @property
    def Y(self):
        # remove the Nt dimension
        return self.Y_st[:, 0, :]

    @property
    def Y_flat(self):
        return self.Y

class MultiOutputTemporalData(SequentialData):
    """
    Within the Kalman Filtering and Smoothing algorithms multi-output temporal data and
        spatio-temporal data are handled in similarilily. However the way the data must be pre-processed is slightly different, therefore we have separate classes between for SpatioTemporalData and MultiOutputTemporalData data.
    """
    def __init__(self, X, Y, sort=True, train_y = False):
        """
        Args:
            X: rank 2 input X if shape N x D
            Y: either rank 2 of shape N x P or rank 3 of shape  Nt X Ns x P = Nt x 1 x P. The extra dimension is for compatability with spatio-temporal multi-output data
        """

        super(MultiOutputTemporalData, self).__init__()

        # Only supports single output 
        chex.assert_rank(X, 2)

        if sort:
            chex.assert_rank(Y, 2)

            X_sorted, Y_sorted = self.sort(
                X, Y
            )

            # self.sort sorts onto a spatio-temporal grid. We only require the temporal part.
            X_sorted = X_sorted[..., 0]

        else:
            X_sorted = X
            Y_sorted = Y   

        chex.assert_rank(X_sorted, 2)
        chex.assert_rank(Y_sorted, 3)

        self._Y = Parameter(np.array(Y_sorted), train=train_y, name='Y')
        self.save_X(X_sorted, train=False)

        # Useful statistcs of the data
        self.Nt = X_sorted.shape[0]
        self.Ns = 1
        self.D = 1
        self.N = self.Ns*self.Nt
        self.output_dim = Y_sorted.shape[-1]
        self.P = self.output_dim


    @property
    def X_space(self):
        return None

    @property
    def X_time(self):
        return self._X.X[:, 0]

    @property
    def Y_st(self):
        return self._Y.value

    @property
    def Y(self):
        # remove the Nt dimension
        return self.Y_st[:, 0, :]

    @property
    def Y_flat(self):
        return self.Y

def get_sequential_data_obj(X, Y, sort):
    if (X.shape[1] == 1) and (Y.shape[1] == 1):
        return TemporalData(X, Y, sort=sort)
    elif (X.shape[1] == 1) and (Y.shape[1] > 1):
        return MultiOutputTemporalData(X, Y, sort=sort) 
    elif X.shape[1] > 1:
        return SpatioTemporalData(X=X, Y=Y, sort=sort) 

    raise RuntimeError()

