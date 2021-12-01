import jax
import jax.numpy as np
import objax
import chex
from .. import settings
from ..utils.utils import ensure_module_list, can_batch
from batchjax import batch_or_loop

from ..transforms  import LinearTransform, NonLinearTransform
from . import ApproximatePosterior, GaussianApproximatePosterior
from ..computation.matrix_ops import vectorized_lower_triangular_cholesky, lower_triangle
import chex
from ..computation.ell_callers import linear_transform_ell, non_linear_transform_ell
from ..computation.predictor_callers import linear_predictor, non_linear_predictor

from typing import Optional, List

def batch_over_posteriors(post_list, fn):
    arr = batch_or_loop(
        fn,
        [post_list],
        [0],
        dim = post_list,
        out_dim = 1,
        batch_flag = can_batch(post_list)
    )
    return arr

class MeanFieldApproximatePosterior(ApproximatePosterior):
    def __init__(self, dim_list: List[int]=None, approximate_posteriors: Optional[List[GaussianApproximatePosterior]]=None):
        super(MeanFieldApproximatePosterior, self).__init__()

        self.num_of_latents = len(dim_list)

        if approximate_posteriors is None:
            self.approx_posteriors = objax.ModuleList([
                GaussianApproximatePosterior(dim=dim_list[q])
                for q in range(self.num_of_latents)
            ])
        else: 
            self.approx_posteriors = approximate_posteriors

    @property
    def m(self):
        m_arr = batch_over_posteriors(
            self.approx_posteriors,
            lambda latent:  latent.m
        )

        return m_arr

    @property
    def S_chol(self):
        arr = batch_over_posteriors(
            self.approx_posteriors,
            lambda latent:  latent.S_chol
        )

        return arr

    @property
    def S(self):
        arr = batch_over_posteriors(
            self.approx_posteriors,
            lambda latent:  latent.S
        )
        return arr

    @property
    def S_diag(self):
        arr = batch_over_posteriors(
            self.approx_posteriors,
            lambda latent:  latent.S_diag
        )
        return arr

    def sample_from_precomputed(self, n_samples):
        N = self.precomputed_marginal_mean_arr[0].shape[0]
        num_latents = len(self.precomputed_marginal_mean_arr)

        samples = objax.random.normal(
            (n_samples,num_latents, N, 1), 
            mean=self.precomputed_marginal_mean_arr[None, ...], 
            stddev = np.sqrt(self.precomputed_marginal_var_arr)[None, ...],
            generator=self.generator
        )

        return samples
