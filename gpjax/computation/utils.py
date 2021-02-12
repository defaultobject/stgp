from ..likelihoods import *
from ..approximate_posteriors import *
from ..distributions import *

from .general import log_chol_matrix_det, cholesky_solve
from ..settings import Settings

import jax
import jax.numpy as np
from jax.config import config
from jax import jit, partial

import typing
from typing import List

def get_reparameterised_lmc_prior(X1_arr:List[np.ndarray], X2_arr:List[np.ndarray], likelihood: Likelihood, prior_arr: List[Distribution]) -> List[np.ndarray]:
    """
        Returns reparametrised kernels of shapes X1xX2
    """
    num_latents = likelihood.num_latents
    num_outputs = likelihood.num_outputs

    #collect mixing weights
    mixing_weights = likelihood.coregion_weights


    #collect kernel blocks
    k_blocks = []
    for output in range(num_outputs):
        print('==========')
        K_output = np.zeros([X1_arr[output].shape[0],  X2_arr[output].shape[0]])
        for latent in range(num_latents):
            k_xx = prior_arr[latent].kernel.K(X1_arr[output], X2_arr[output])
            k_xx = k_xx * mixing_weights[output, latent]**2
            print(mixing_weights[output, latent]**2, k_xx)
            K_output += k_xx
        print(K_output)
        k_blocks.append(K_output)

    return k_blocks

def get_blocks_from_lmc_proir(X1_arr:List[np.ndarray], X2_arr:List[np.ndarray], likelihood: Likelihood, prior_arr: List[Distribution]) -> List[np.ndarray]:
    """
        Returns reparametrised kernels of shapes X1xX2
    """
    num_latents = likelihood.num_latents
    num_outputs = likelihood.num_outputs

    #collect mixing weights
    mixing_weights = likelihood.coregion_weights


    #collect kernel blocks
    #assume that X1 X2 are the inputs for each latent function
    k_blocks = []
    for latent in range(num_latents):
        k_xx = prior_arr[latent].kernel.K(X1_arr[0], X2_arr[0])
        k_blocks.append(k_xx)

    return jax.scipy.linalg.block_diag(*k_blocks)

def get_blocks_from_lmc_proir_diagional(X1_arr:List[np.ndarray], likelihood: Likelihood, prior_arr: List[Distribution]) -> List[np.ndarray]:
    """
        Returns reparametrised kernels of shapes X1xX2
    """
    num_latents = likelihood.num_latents
    num_outputs = likelihood.num_outputs

    #collect mixing weights
    mixing_weights = likelihood.coregion_weights


    #collect kernel diagional vectors
    k_blocks = []
    for latent in range(num_latents):
        k_xx = prior_arr[latent].kernel.K_diag(X1_arr[latent])
        k_blocks.append(k_xx)

    return np.hstack(k_blocks)[:, None]

def get_reparameterised_lmc_prior_diagional(X1_arr:List[np.ndarray], likelihood: Likelihood, prior_arr: List[Distribution]) -> List[np.ndarray]:
    """
        Returns reparametrised kernels of shapes X1xX2
    """
    num_latents = likelihood.num_latents
    num_outputs = likelihood.num_outputs

    #collect mixing weights
    mixing_weights = likelihood.coregion_weights 

    #collect kernel blocks
    k_blocks = []
    for output in range(num_outputs):
        K_output = None
        for latent in range(num_latents):
            k_xx = prior_arr[latent].kernel.K_diag(X1_arr[output])
            k_xx = k_xx * mixing_weights[output, latent]**2
            if K_output is None:
                K_output = k_xx
            else:
                K_output += k_xx
        k_blocks.append(K_output)

    return k_blocks
