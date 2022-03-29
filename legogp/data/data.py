import objax
import chex
import jax.numpy as np
import numpy as onp

from .sequential import order_sequentially_np, pad_with_nan_to_make_grid
from .. import Parameter

class Input(objax.Module):
    def __init__(self, X, name='X', train=False):

        if isinstance(X, Input):
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

        return self._X_ref.X

class Data(objax.Module):
    def __init__(self, X, Y):
        #chex.assert_rank(X, 2)
        #chex.assert_rank(Y, 2)
        #chex.assert_equal(X.shape[0], Y.shape[0])

        self._Y = Parameter(np.array(Y), train=False, name='Y')
        self._X = Input(X, train=False)

        self.N = Y.shape[0]

    @property
    def Y(self):
        return self._Y.value

    @property
    def X(self):
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


class AggregatedData(Data):
    pass

class SpatialAggregatedData(AggregatedData):
    pass

class TemporalAggregatedData(AggregatedData):
    pass

class SequentialData(Data):
    def __init__(self):
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


        points_added, X_padded, Y_padded = pad_with_nan_to_make_grid(
            X,
            Y
        )

        unique_idx, reverse_unique_idx, sort_idx, X_sorted, Y_sorted = order_sequentially_np(
            X_padded, Y_padded
        )


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
    def Y(self):
        return self._x_time.value

    def unsort(self, A):
        return A[self.sort_idx][self.reverse_unique_idx][:self.num_original_points]

    @property
    def Y_flat(self):
        raise NotImplementedError()


class SpatioTemporalData(SequentialData):
    def __init__(self, X_time = None, X_space = None, X = None,  Y = None, sort=True):
        """

        There are two cases supported:

        1) X is already sorted 

            X_time: Nt  
            X_space: Ns x D
            X: None
            Y: Nt x Ns x P

        2) X is not sorted 

            X_time: None
            X_space: None
            X: N x 1 
            Y: N x P

        """

        if sort:
            # X_time and X_space are None
            if X is None: raise RuntimeError('X must be passed')

            chex.assert_rank([X, Y], [2, 2])

            X_sorted, Y_sorted = self.sort(X, Y)

            # self.sort returns the full spatio-temporal dataset but we only require the temporal
            #  and spatial parts
            X_time = X_sorted[:, 0, 0]
            X_space = X_sorted[0, :, 1:]
            Y = Y_sorted

        chex.assert_rank(X_time, 1)
        chex.assert_rank(X_space, 2)
        chex.assert_rank(Y, 3)
        chex.assert_equal(X_time.shape[0], Y.shape[0])
        chex.assert_equal(X_space.shape[0], Y.shape[1])

        # Y is in time - space order
        self._Y = Parameter(np.array(Y), train=False, name='Y')
        self._x_time = Parameter(np.array(X_time), train=False, name='X_time')
        self._X_space = Parameter(np.array(X_space), train=False, name='X_space')

        # Useful statistcs of the data
        self.Nt = X_sorted.shape[0]
        self.Ns = X_space.shape[0]
        self.D = X_space.shape[1]+1
        self.N = self.Ns*self.Nt
        self.P = Y.shape[2]

    @property
    def X(self):
        """ Constructs the full spatio-temporal input from the temporal and spatial parts. """

        # We return X so that it is already ordered.
        #  ie in time-space ordering
        X_t = np.repeat(self.X_time[:, None], self.Ns)[:, None]
        X_s = np.tile(self.X_space, [self.Nt, 1])

        X = np.hstack([X_t, X_s])

        return X

    @property
    def X_space(self):
        return self._X_space.value

    @property
    def X_time(self):
        return self._x_time.value

    @property
    def Y(self):
        return self._Y.value

    @property
    def Y_flat(self):
        # X is returned in time-space ordering  so we return Y in the same order
        # Y is already sorted by time, and then by space.
        # Therefore all we have to is reshape

        Y = np.reshape(
            self.Y,
            [-1, self.P]
        )

        return Y


class TemporalData(SequentialData):
    def __init__(self, X_time, Y, sort=True):
        """
        There are two cases supported:

        1) X is already sorted 

            X_time: Nt x 1 
            Y: Nt x 1 x 1

        2) X is not sorted 

            X_time: Nt x 1 
            Y: Nt x 1 
        """

        super(TemporalData, self).__init__()

        # Only supports single output 
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

        self._Y = Parameter(np.array(Y_sorted), train=False, name='Y')
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
    def Y(self):
        return self._Y.value

    @property
    def Y_flat(self):
        return self.Y

class MultiOutputTemporalData(SequentialData):
    """
    Within the Kalman Filtering and Smoothing algorithms multi-output temporal data and
        spatio-temporal data are handled in similarilily. However the way the data must be pre-processed is slightly different, therefore we have separate classes between for SpatioTemporalData and MultiOutputTemporalData data.
    """
    def __init__(self, X_time, Y, sort=True):

        super(MultiOutputTemporalData, self).__init__()

        # Only supports single output 
        chex.assert_rank(X_time, 2)

        if sort:
            chex.assert_rank(Y, 2)

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

        self._Y = Parameter(np.array(Y_sorted), train=False, name='Y')
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
    def Y(self):
        return self._Y.value

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

