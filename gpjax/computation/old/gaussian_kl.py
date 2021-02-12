from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as np

from .general import log_chol_matrix_det, cholesky_solve
from ..settings import Settings
from jax import jit


@jit
def gaussian_kl(mu_1, covar_chol_1, mu_2, covar_chol_2):
    covar_1 = covar_chol_1 @  covar_chol_1.T
    covar_2 = covar_chol_2 @  covar_chol_2.T

    log_det_term = log_chol_matrix_det(covar_chol_2) - log_chol_matrix_det(covar_chol_1)

    #trace_term = np.trace(cholesky_solve(covar_chol_2, covar_1))
    trace_term = np.sum(np.square(jax.scipy.linalg.solve_triangular(covar_chol_2, covar_chol_1, lower=True)))

    err = mu_2-mu_1 
    maha_term = np.matmul(err.T, cholesky_solve(covar_chol_2, err))
    maha_term = np.sum(maha_term)

    N = mu_1.shape[0]*1.0

    return 0.5 * (log_det_term - N + trace_term + maha_term )

@jit
def whitened_gaussian_kl(mu_1, covar_chol_1, mu_2, covar_chol_2):
    covar_1 = covar_chol_1 @  covar_chol_1.T

    log_det_term = - log_chol_matrix_det(covar_chol_1)

    trace_term = np.trace(covar_1)

    maha_term = np.sum(np.square(mu_1))

    N = mu_1.shape[0]*1.0

    return 0.5 * (log_det_term - N + trace_term + maha_term )
