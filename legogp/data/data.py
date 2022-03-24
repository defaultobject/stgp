import objax
import chex
import jax.numpy as np
import numpy as onp

from .sequential import order_sequentially_np, pad_with_nan_to_make_grid
from .. import Parameter

class Input(objax.Module):
    def __init__(self, X, train=False):

        if isinstance(X, Input):
            self._X_ref = X # Store reference
            self._X = None
        else:
            self._X = Parameter(np.array(X), train=train, name='X')
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

class AggregatedData(Data):
    pass

class SpatialAggregatedData(AggregatedData):
    pass

class TemporalAggregatedData(AggregatedData):
    pass

class SequentialData(Data):
    pass


class SpatioTemporalData(SequentialData):
    def __init__(self, X_time = None, X_space = None, X = None,  Y = None, sort=True):
        """
            X_time: Nt x 1
            X_space: Ns x D
            Y: Nt x Ns x P
        """

        if sort:
            # X_time and X_space are None
            original_points = X.shape[0]
            points_added, X_padded, Y_padded = pad_with_nan_to_make_grid(
                X,
                Y
            )

            unique_idx, sort_idx, X_sorted, Y_sorted = order_sequentially_np(
                X_padded, Y_padded
            )

            X_time = X_sorted[:, 0, 0]
            X_space = X_sorted[0, :, 1:]
            Y = Y_sorted


        chex.assert_rank(X_time, 1)
        chex.assert_rank(X_space, 2)
        chex.assert_rank(Y, 3)
        chex.assert_equal(X_time.shape[0], Y.shape[0])
        chex.assert_equal(X_space.shape[0], Y.shape[1])

        # Store original np arrays for prediction time
        self._X_time_np = X_time
        self._X_space_np = X_space
        self._Y_np = Y_sorted

        # Y is in time - space order
        self._Y = Parameter(np.array(Y), train=False, name='Y')
        self._x_time = Parameter(np.array(X_time), train=False, name='X_time')
        self._X_space = Parameter(np.array(X_space), train=False, name='X_space')

        # Useful statistcs of the data
        self.Nt = X_sorted.shape[0]
        self.Ns = X_space.shape[0]
        self.D = X_space.shape[1]+1
        self.N = self.Ns*self.Nt

        # Required to under the sorting step
        self.unique_idx = unique_idx
        self.sort_idx = sort_idx
        self.points_added = points_added
        self.original_points = original_points

    @property
    def X_np(self):
        # X must be in time - space order
        stacked_time_points = onp.repeat(self._X_time_np, self.Ns)[:, None]
        stacked_space_points = onp.vstack(onp.tile(
            self._X_space_np,
            [self.Nt, 1, 1]
        ))
        X = onp.hstack([stacked_time_points, stacked_space_points])
            
        return X

    @property
    def Y_np(self):
        return np.vstack(self._Y_np)

    @property
    def X_space(self):
        return self._X_space.value

    @property
    def X_time(self):
        return self._x_time.value

    @property
    def X_space(self):
        return self._X_space.value

    @property
    def Y(self):
        return self._Y.value

    def unsort(self, A):
        return A[self.sort_idx][self.unique_idx][:self.original_points]

#self.sort_idx = sort_idx
#self.unique_idx = unique_idx
class TemporalData(SequentialData):
    def __init__(self, X_time, Y, sort=True):

        if sort:
            # Sort
            chex.assert_rank(X_time, 2)
            chex.assert_rank(Y, 2)

            original_points = X_time.shape[0]

            points_added, X_padded, Y_padded = pad_with_nan_to_make_grid(
                X_time,
                Y
            )

            unique_idx, sort_idx, X_sorted, Y_sorted = order_sequentially_np(
                X_padded, Y_padded
            )

            # Remove spatial dimension
            X_sorted = X_sorted[..., 0]

        else:
            chex.assert_rank(X_time, 2)
            chex.assert_rank(Y, 3)

            X_sorted = X_time
            Y_sorted = Y   

            unique_idx = None
            sort_idx = None
            points_added = None
            original_points = None

        chex.assert_rank(X_sorted, 2)
        chex.assert_rank(Y_sorted, 3)

        # Store data as objax parameters so that they can be used on GPUs etc
        if isinstance(X_time, Input):
            self._X = X_time # Store as reference
        else:
            self._X = Input(np.array(X_sorted), train=False)

        self._Y = Parameter(np.array(Y_sorted), train=False, name='Y')

        # Useful statistcs of the data
        self.Nt = X_sorted.shape[0]
        self.Ns = 1
        self.D = 1
        self.N = self.Ns*self.Nt

        # Required to under the sorting step
        self.unique_idx = unique_idx
        self.sort_idx = sort_idx
        self.points_added = points_added
        self.original_points = original_points

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

    def unsort(self, A):
        return A[self.sort_idx][self.unique_idx][:self.original_points]



