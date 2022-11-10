import jax.numpy as np
import objax
from typing import List, Optional
from batchjax import batch_or_loop

from . import ApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior

from ..likelihood import DiagonalGaussian, ProductLikelihood, BlockDiagonalGaussian
from ..utils.utils import get_batch_type

import numpy as onp


class ConjugateApproximatePosterior(ApproximatePosterior):
    pass

class ConjugateGaussian(GaussianApproximatePosterior, ConjugateApproximatePosterior):
    def __init__(self, X, block_size: int, num_blocks:int = None, Y_tilde = None, surrogate_model: 'Model' = None):
        """
        A conjugate gaussian represents the approximate posterior as:
            q(u) \propto N(Y_tilde | u, V_tilde) p(u)
        For computational reasons it is generally more efficient to store V_tilde latent-data format. 
        For consistency we also store Y_tilde in the same way.
        """

        if X is None:
            raise RuntimeError('X must be passed')

        self.dim = X.shape[0]

        self.block_size = block_size

        if num_blocks is None:
            self.num_blocks = int(self.dim/self.block_size)
        else:
            self.num_blocks = num_blocks
            
        if Y_tilde is not None:
            Y_tilde = np.array(Y_tilde)
        else:
            Y_tilde = 1e-5*np.ones([self.num_blocks, self.block_size, 1])

        V_tilde = np.tile(np.eye(self.block_size), [self.num_blocks, 1, 1])

        surrogate_likelihood = [BlockDiagonalGaussian(
            block_size=self.block_size,
            num_blocks = self.num_blocks,
            variance=V_tilde
        )]

        self.surrogate = surrogate_model(
            X = X,
            Y = Y_tilde,
            likelihood = surrogate_likelihood
        )

    def fix(self):
        self.surrogate.likelihood.fix()

class DiagonalConjugateGaussian(ConjugateApproximatePosterior):
    def __init__(self, dim: int=None, surrogate_model: 'Model' = None):

        if dim is None :
            raise RuntimeError('Dim must be passed')

        self.dim = dim
        self.surrogate = surrogate_model

        # TODO: Nat params


class BlockDiagonalConjugateGaussian(ConjugateApproximatePosterior):
    def __init__(self, dim: int=None, surrogate_model: 'Model' = None):

        if dim is None :
            raise RuntimeError('Dim must be passed')

        self.dim = dim
        self.surrogate = surrogate_model

        # TODO: Nat params

class MeanFieldConjugateGaussian(ConjugateApproximatePosterior, MeanFieldApproximatePosterior):
    def __init__(self, approximate_posteriors: Optional[List[ConjugateGaussian]]=None):

        if approximate_posteriors is None:
            raise RuntimeError()

        elif type(approximate_posteriors) is list: 
            self.approx_posteriors = objax.ModuleList(approximate_posteriors)
        else:
            self.approx_posteriors = approximate_posteriors

    @property
    def Y(self):
        q_list = self.approx_posteriors
        Y_arr =  batch_or_loop(
            lambda q: q.surrogate.Y,
            [q_list],
            [0],
            dim=len(q_list),
            out_dim = 1,
            batch_type = get_batch_type(q_list)
        )
        # Fix shapes

        return Y_arr[..., 0].T

    @property
    def X(self):
        q_list = self.approx_posteriors
        return  batch_or_loop(
            lambda q: q.surrogate.X,
            [q_list],
            [0],
            dim=len(q_list),
            out_dim = 1,
            batch_type = get_batch_type(q_list)
        )

    @property
    def likelihood(self):
        # TODO: check this
        return ProductLikelihood([
            q.surrogate.likelihood.likelihood_arr[0] for q in self.approx_posteriors
        ])

class FullConjugateGaussian(ConjugateGaussian, FullGaussianApproximatePosterior):
    def __init__(self, X, num_latents: int, block_size: int, surrogate_model: 'Model' = None, num_blocks: int = None):
        """
        A conjugate gaussian represents the approximate posterior as:

            q(U) \propto N(Y_tilde | U, V_tilde) p(U)

        For computational reasons it is generally more efficient to store V_tilde (data-latent or time-latent-space) format.  For consistency we also store Y_tilde in the same way. Leading to Y_tilde and V_tilde havning shapes:
            
            Y_tilde: [N x num_latents] or [Nt x (num_latents * Ns)]
            V_tilde: [N x block_size x block_size] or [Ns x (block_size * Ns) x (block_size * Ns)]

        There are three main use cases:
            1) No Sparsity 
            2) Full Sparsity
            3) Spatial Sparsity

        In the case of no sparsity the approximate likelihood will be block diagonal. There will be N blocks each of size Q x Q, which captures the correlation between latent processes.

        In the case of full sparsity the approximate likelihood will be dense.
        """

        self.block_size = block_size
        self.num_latents = num_latents

        if num_blocks is None:
            self.M = X.shape[0]
        else:
            self.M = num_blocks

        if num_blocks is None:
            self.num_blocks = int((self.num_latents*self.M)/block_size)
        else:
            self.num_blocks = num_blocks

        Y_tilde = 1e-5*np.ones([self.M, self.block_size])*onp.random.rand(self.M, self.block_size)
        Y_tilde = 1e-5*np.ones([self.M, self.block_size])

        V_tilde = np.tile(
            np.ones([self.block_size, self.block_size]) + 2*np.eye(self.block_size), 
            [self.num_blocks, 1, 1]
        )

        surrogate_likelihood = BlockDiagonalGaussian(
            block_size=self.block_size,
            num_blocks = self.num_blocks,
            num_latents = self.num_latents,
            variance=V_tilde
        )

        self.surrogate = surrogate_model(
            X = X,
            Y = Y_tilde,
            likelihood = surrogate_likelihood
        )

    @property
    def likelihood(self):
        return self.surrogate.likelihood

    @property
    def X(self):
        return self.surrogate.X 

    @property
    def Y(self):
        return self.surrogate.Y
