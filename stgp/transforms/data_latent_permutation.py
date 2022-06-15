"""
Converts a prior from latent-data format to data-latent format.  This is required when using a FullApproximatePosterior 

There are two separate classes and we can exploit sparsity when permuting independent priors.
"""
import jax
import jax.numpy as np
from batchjax import batch_or_loop, BatchType
import chex

from . import Transform
from ..computation.matrix_ops import block_from_mat, v_get_block_diagonal
from ..computation.permutations import data_order_to_output_order
from ..utils.utils import ensure_module_list, get_batch_type

class DataLatentPermutationFromFull(Transform):
    def __init__(self, latent):
        self._latent_obj = latent
        self._output_dim = self.latent_obj.output_dim 

    def np_mean(self, X):
        return self.latent_obj.mean(X[0])

    def np_full_covar(self, X1, X2):
        #TODO: why?
        return self.latent_obj.covar(X1[0], X2[0])



class DataLatentPermutation(Transform):
    """
    Converts a prior from latent-data format to data-latent format.
    This is required when using a FullApproximatePosterior .

    Assumes that latent_obj is independent

    Function name syntax:
        p: permute
        lp: left permute
        np: no permute
    """
    def __init__(self, latents):
        # Allow passing a list of prior models and transformed model
        if type(latents) is list:
            self._latent_obj = Independent(latents=latents, prior=True)
        else:
            self._latent_obj = latents 

        self.permutation_fn = data_order_to_output_order
        self._output_dim = self.latent_obj.output_dim 

    @property
    def num_latents(self):
        self.latent_obj.num_latents

    def get_sparsity_list(self):
        return self.latent_obj.get_sparsity_list()

    def get_Z(self):
        return self.latent_obj.get_Z()

    def forward(self, *args, **kwargs):
        return self.latent_obj.forward(*args, **kwargs)

    @property
    def _latents_arr(self):
        return self.latent_obj.latents

    def _mean_blocks(self, X):
        return batch_or_loop(
            lambda  x, latent: latent.mean(x)[0],
            [X, self.latent_obj.latents],
            [0, 0 ],
            dim = self.latent_obj.num_latents,
            out_dim=1,
            batch_type = get_batch_type(self.latent_obj.latents)
        )

    def s_mean_blocks(self, X):
        return batch_or_loop(
            lambda  x, latent: latent.mean(x)[0],
            [X, self.latent_obj.latents],
            [None, 0 ],
            dim = self.latent_obj.num_latents,
            out_dim=1,
            batch_type = get_batch_type(self.latent_obj.latents)
        )

    def mean(self, X):
        return self.permute_vec_blocks(self._mean_blocks(X))

    def s_mean(self, X):
        return self.permute_vec_blocks(self.s_mean_blocks(X))

    def np_mean(self, X):
        return np.vstack(self._mean_blocks(X))

    def permute_vec(self, v):
        P = self.permutation_fn(
            self.num_latents,
            int(v.shape[0]/self.num_latents)
        )

        return P @ v

    def permute_vec_blocks(self, v_blocks):
        # TODO: check this, atm makes no difference as we are zero mean
        return np.reshape(v_blocks, [-1, v_blocks.shape[-1]])

    def permute_blocks(self, A_blocks):
        lp_A =  self.lp_blocks(A_blocks) 

        right_P = self.permutation_fn(
            self.num_latents,
            int(lp_A.shape[-1]/self.num_latents)
        )

        return lp_A @ right_P.T


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

    def _blocks(self, K_blocks):
        """ Block diag without permutation"""
        chex.assert_rank(K_blocks, 3)
        return jax.scipy.linalg.block_diag(*K_blocks)

    def lp_blocks(self, K_blocks):
        chex.assert_rank(K_blocks, 3)

        Q = K_blocks.shape[0]
        N1 = K_blocks.shape[1]

        K = self._blocks(K_blocks)
        return np.vstack(np.transpose(np.reshape(K, [Q, N1, -1]), [1, 0, 2]))

    def p_blocks(self, K_blocks):
        return self.permute_blocks(K_blocks)


    def ls_full_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """ X1 is static. No Permutations """
        K = batch_or_loop(
            lambda x1, x2, latent: latent.covar(x1, x2)[0],
            [X1, X2, self.latent_obj.latents],
            [None, 0, 0],
            dim = self.latent_obj.num_latents,
            out_dim=1,
            batch_type = get_batch_type(self.latent_obj.latents)
        )
        return K

    def s_full_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """ X1 and X2 are static. No Permutations """
        K = batch_or_loop(
            lambda x1, x2, latent: latent.covar(x1, x2)[0],
            [X1, X2, self.latent_obj.latents],
            [None, None, 0],
            dim = self.latent_obj.num_latents,
            out_dim=1,
            batch_type = get_batch_type(self.latent_obj.latents)
        )
        return K

    def _full_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """ No Permutations """
        K = batch_or_loop(
            lambda x1, x2, latent: latent.covar(x1, x2)[0],
            [X1, X2, self.latent_obj.latents],
            [0, 0, 0],
            dim = self.latent_obj.num_latents,
            out_dim=1,
            batch_type = get_batch_type(self.latent_obj.latents)
        )
        return K

    def lp_ls_full_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Left Permute, keep X1 static when constructing full var
        """
        return self.lp_blocks(self.ls_full_covar(X1, X2))

    def lp_s_full_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Left Permute, keep both x1 and x2 static when constructing full var
        """
        return self.lp_blocks(self.s_full_covar(X1, X2))

    def lp_full_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Left Permute
        """
        return self.lp_blocks(self._full_covar(X1, X2))

    def p_s_full_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Permute, static X1 and X2
        """
        return self.p_blocks(self.s_full_covar(X1, X2))

    def p_full_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Permute
        """
        return self.p_blocks(self._full_covar(X1, X2))

    def np_full_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        No Permute
        """
        return self._blocks(self._full_covar(X1, X2))

    def full_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.p_full_covar(X1, X2)


    def vec_var(self, X1: np.ndarray) -> np.ndarray:
        K = self.latent_obj.var(X1)
        K = np.vstack(K)
        return self.permute_vec(K)

    def full_var(self, X1: np.ndarray) -> np.ndarray:
        return self.covar(X1, X1)

    def s_blocks_var(self, X: np.ndarray, group_size, block_size) -> np.ndarray:
        raise NotImplementedError()

    def blocks_var(self, X: np.ndarray, group_size, block_size) -> np.ndarray:
        """
        X is a grouped input matrix:
            [N_g, S_g, D]
        group_size is required data_grouping
        block_size is the size variance for the corresponding groups
        """
        # Group data

        X = jax.vmap(
            block_from_mat,
            [0, None],
            0
        )(X, group_size)

        X = np.transpose(X, [1, 0, 2, 3])


        # For each group collect blocks
        K_blocks = jax.vmap(
            lambda m, x: m.p_full_covar(x, x),
            [None, 0],
            0
        )(self, X)

        K_blocks = v_get_block_diagonal(
            K_blocks,
            block_size,
            K_blocks.shape[1]
        )

        return K_blocks
