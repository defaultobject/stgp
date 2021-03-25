from ..settings import jitter
from ..kernel import Kernel, RBF
from ..likelihood import Gaussian
from ..approximate_posteriors import GaussianApproximatePosterior
from ..dispatch import dispatch
from .gaussian import log_gaussian
from ..batching import Batched
from .. import utils
from .matrix_ops import cholesky, log_chol_matrix_det, add_jitter

import jax
from jax import jit
import jax.numpy as np
import chex
from typing import List
from objax import ModuleList


@dispatch(object, object, GaussianApproximatePosterior, Gaussian, object, object)
def predict_diagonal(XS, X,  approximate_posterior, likelihood, kernel, sparsity):
    m, S_diag = approximate_posterior.predictive_marginal(XS, X, kernel, sparsity)
    return m, S_diag + likelihood.variance

