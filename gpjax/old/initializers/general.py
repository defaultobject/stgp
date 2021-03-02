import numpy as np


def vectorized_lower_triangular_cholesky(A: np.ndarray) -> np.ndarray:
    """
    Takes the cholesky decomposition of A vectorized the output
    """
    N = A.shape[0]
    init = np.linalg.cholesky(A) + 1e-7 * np.eye(
        N
    )  # add jitter for numerical stability
    init = init[np.tril_indices(N)].flatten()
    return init
