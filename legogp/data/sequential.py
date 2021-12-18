""" Numpy operations for dealing sequential data """
import numpy as np

def order_sequentially(X, Y = None):
    """
        lexsort uses the final column as the primary sort key and then sorts by each column from the last
            1 -  roll the columns so the time is the last column
            2 - get idx of new ordering
            3 - roll X so that time is the first axis again

        NOTE: assumes that X can be represented as a spatio-temporal grid
    """
    assert len(X.shape) == 2

    if Y is not None:
        assert X.shape[0] == Y.shape[0]

    time_zero = X[0, 0]

    # Since X is a spatio-temporal grid we can just extract the spatial points at the first time
    X_spatial = X[X[:, 0]==time_zero][:,1:]

    grid_size = X_spatial.shape[0]
    time_points = int(X.shape[0]/grid_size)

    # Put time axis as last axis os that this is sorted first
    #  it does not matter the order that the spatial dimensions get sorted
    X = np.roll(X, -1, axis=1)

    # Get index to sort by time points and then spatial points
    #   we need the index so that we can undo the sort later
    idx = np.lexsort(X.T)

    # Sort
    X = X[idx]

    if Y is not None:
        Y = Y[idx]

    #reset time axis
    X = np.roll(X, 1, axis=1)

    #reshape for grid structure
    X = np.reshape(X, [time_points, grid_size, X.shape[1]])

    if Y is not None:
        Y = np.reshape(Y, [time_points, grid_size, 1])

        return idx, X, Y

    return idx, X
