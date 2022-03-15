import jax
import jax.numpy as np
from batchjax import batch_or_loop, BatchType

from . import Transform
from ..computation.matrix_ops import block_from_mat, v_get_block_diagonal
from ..utils.utils import ensure_module_list, get_batch_type

class Permutation(Transform):
    def __init__(self, latents, permutation_fn=None):
        # Allow passing a list of prior models and transformed model
        if type(latents) is list:
            self._latent_obj = Independent(latents=latents, prior=True)
        else:
            self._latent_obj = latents 

        if permutation_fn is None:
            raise RuntimeError('A permutation function must be passed')

        # TODO: this will actually only work for one type of permutation
        self.permutation_fn = permutation_fn

        self.num_latents = self.latent_obj.num_latents

    def get_sparsity_list(self):
        return self.latent_obj.get_sparsity_list()

    def permute_vec(self, v):
        P = self.permutation_fn(
            self.num_latents,
            int(v.shape[0]/self.num_latents)
        )

        return P @ v

    def permute_mat(self, A):
        left_P = self.permutation_fn(
            self.num_latents,
            int(A.shape[0]/self.num_latents)
        )

        right_P = self.permutation_fn(
            self.num_latents,
            int(A.shape[1]/self.num_latents)
        )

        return left_P @ A @ right_P.T

    def unpermute_vec(self, v):
        P = self.permutation_fn(
            self.num_latents,
            int(v.shape[0]/self.num_latents)
        )

        return P.T @ v

    def unpermute_mat(self, A):
        left_P = self.permutation_fn(
            self.num_latents,
            int(A.shape[0]/self.num_latents)
        )

        right_P = self.permutation_fn(
            self.num_latents,
            int(A.shape[1]/self.num_latents)
        )

        return left_P.T @ A @ right_P


    def vec_mean(self, X1: np.ndarray) -> np.ndarray:
        m = batch_or_loop(
            lambda x1, latent: latent.mean(x1)[0],
            [X1, self.latent_obj.latents],
            [0, 0],
            dim = self.latent_obj.num_latents,
            out_dim=1,
            batch_type = get_batch_type(self.latent_obj.latents)
        )
        m = np.vstack(m)

        return self.permute_vec(m)

    def full_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        X1 and X2 are grouped, one per latent function
        """

        K = batch_or_loop(
            lambda x1, x2, latent: latent.covar(x1, x2)[0],
            [X1, X2, self.latent_obj.latents],
            [0, 0, 0],
            dim = self.latent_obj.num_latents,
            out_dim=1,
            batch_type = get_batch_type(self.latent_obj.latents)
        )

        K = jax.scipy.linalg.block_diag(*K)

        return self.permute_mat(K)


    def vec_var(self, X1: np.ndarray) -> np.ndarray:
        K = self.latent_obj.var(X1)
        K = np.vstack(K)
        return self.permute_vec(K)

    def full_var(self, X1: np.ndarray) -> np.ndarray:
        return self.covar(X1, X1)

    def blocks_var(self, X: np.ndarray, group_size, block_size) -> np.ndarray:
        """
        X is a grouped input matrix:
            [N_g, S_g, D]
        group_size is required data_grouping
        block_size is the size variance for the corresponding groups
        """
        # Group data

        #X = np.hstack(X)
        X = jax.vmap(
            block_from_mat,
            [0, None],
            0
        )(X, group_size)

        X = np.transpose(X, [1, 0, 2, 3])


        # For each group collect blocks
        K_blocks = jax.vmap(
            lambda m, x: m.full_covar(x, x),
            [None, 0],
            0
        )(self, X)

        K_blocks = v_get_block_diagonal(
            K_blocks,
            block_size,
            K_blocks.shape[1]
        )

        return K_blocks
