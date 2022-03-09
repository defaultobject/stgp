import jax
import jax.numpy as np

from . import Transform

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
        m = self.latent_obj.mean(X1)
        m = np.vstack(m)

        return self.permute_vec(m)

    def full_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        K = self.latent_obj.covar(X1, X2)
        K = jax.scipy.linalg.block_diag(*K)

        return self.permute_mat(K)


    def vec_var(self, X1: np.ndarray) -> np.ndarray:
        K = self.latent_obj.var(X1)
        K = np.vstack(K)
        return self.permute_vec(K)

    def full_var(self, X1: np.ndarray) -> np.ndarray:
        return self.covar(X1, X1)

    def blocks_var(self, X1: np.ndarray) -> np.ndarray:
        K_blocks = jax.vmap(
            lambda m, x: m.full_covar(x[:, None], x[:, None]),
            [None, 0],
            0
        )(self, X1)

        return K_blocks

