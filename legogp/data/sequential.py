""" Numpy operations for dealing sequential data """
import numpy as np

def pad_with_nan_to_make_grid(X, Y):
    #converts data into grid

    N = X.shape[0]

    #construct target grid
    unique_time = np.unique(X[:, 0])
    unique_space = np.unique(X[:, 1:], axis=0)

    Nt = unique_time.shape[0]
    Ns = unique_space.shape[0]

    print('grid size:', N, Nt, Ns, Nt*Ns)

    X_tmp = np.tile(np.expand_dims(unique_space, 0), [Nt, 1, 1])

    time_tmp = np.tile(unique_time, [Ns]).reshape([Nt, Ns], order='F')

    X_tmp = X_tmp.reshape([Nt*Ns, -1])

    time_tmp = time_tmp.reshape([Nt*Ns, 1])

    #X_tmp is the full grid
    X_tmp = np.hstack([time_tmp, X_tmp])

    #Find the indexes in X_tmp that we need to add to X to make a full grid
    _X = np.vstack([X,  X_tmp])
    _Y = np.nan*np.zeros([_X.shape[0], 1])

    _, idx = np.unique(_X, return_index=True, axis=0)
    idx = idx[idx>=N]
    print('unique points: ', idx.shape)

    X_to_add = _X[idx, :]
    Y_to_add = _Y[idx, :]

    X_grid = np.vstack([X, X_to_add])
    Y_grid = np.vstack([Y, Y_to_add])

    #sort for good measure
    _X = np.roll(X_grid, -1, axis=1)
    #sort by time points first
    idx = np.lexsort(_X.T)

    return X_grid[idx], Y_grid[idx]

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

    # Get unique rows (time, space, features) to remove duplicates
    _, unique_idx, reverse_idx = np.unique(X, axis=0, return_index = True, return_inverse=True)

    # Get index to sort by time points and then spatial points
    #   we need the index so that we can undo the sort later
    # required so that all spatial points are consistenly organised

    # Get unique points
    X = X[unique_idx]

    # Since X is a spatio-temporal grid we can just extract the spatial points at the first time
    X_spatial = X[X[:, 0]==time_zero][:,1:]

    # Put time axis as last axis os that this is sorted first
    #  it does not matter the order that the spatial dimensions get sorted
    X = np.roll(X, -1, axis=1)

    grid_size = X_spatial.shape[0]
    time_points = int(X.shape[0]/grid_size)

    # Sort in space and time
    idx = np.lexsort(X.T)

    X = X[idx]

    if Y is not None:
        Y = Y[unique_idx][idx]

    #reset time axis
    X = np.roll(X, 1, axis=1)

    #reshape for grid structure
    X = np.reshape(X, [time_points, grid_size, X.shape[1]])


    if Y is not None:
        Y = np.reshape(Y, [time_points, grid_size, 1])

        return reverse_idx, idx, X, Y

    return reverse_idx, idx, X
