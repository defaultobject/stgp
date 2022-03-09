import jax.numpy as np
import objax
from typing import List, Optional
from batchjax import batch_or_loop

from . import ApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior

from ..likelihood import DiagonalGaussian, ProductLikelihood, BlockDiagonalGaussian
from ..utils.utils import get_batch_type


class ConjugateApproximatePosterior(ApproximatePosterior):
    pass

class ConjugateGaussian(GaussianApproximatePosterior, ConjugateApproximatePosterior):
    def __init__(self, X, surrogate_model: 'Model' = None):

        if X is None :
            raise RuntimeError('X must be passed')

        self.dim = X.shape[0]

        Y_tilde = 1e-5*np.ones([self.dim, 1])
        V_tilde = np.ones([self.dim, 1])

        surrogate_likelihood = DiagonalGaussian(
            V_tilde
        )

        self.surrogate = surrogate_model(
            X = X,
            Y = Y_tilde,
            likelihood = surrogate_likelihood
        )

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
    def __init__(self, X, num_latents: int, block_size: int, surrogate_model: 'Model' = None):

        self.block_size = block_size
        self.num_latents = num_latents
        self.num_blocks = int((self.num_latents*X.shape[0])/block_size)

        Y_tilde = 1e-5*np.ones([self.num_blocks, self.block_size])
        V_tilde = np.tile(np.eye(self.block_size), [self.num_blocks, 1, 1])

        if True:
            V_tilde = np.tile(
                np.ones([self.block_size, self.block_size]) + 2*np.eye(self.block_size), 
                [self.num_blocks, 1, 1]
            )

        surrogate_likelihood = BlockDiagonalGaussian(
            block_size=self.block_size,
            num_blocks = self.num_blocks,
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
