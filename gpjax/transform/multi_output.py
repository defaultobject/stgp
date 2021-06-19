"""Multi-output/task specific transforms."""
from .transform import Transform, LinearTransform, NonLinearTransform
import typing
from typing import List, Optional
import jax.numpy as np
import objax
from ..batching import batch


class LMC(LinearTransform):
    r"""
    Linear model of coregionilisation.

    Generative model:
        f_p = \sum w_{p,q} g_q

    """
    class LMC_p(LinearTransform):
        """
            Individual transofrm for task p
        """
        def __init__(self, latents_p, W_p):
            self.latents_p = latents_p
            self.raw_W_p = objax.TrainVar(W_p)

        @batch
        def W_p(self, raw_getter):
            return raw_getter()

        def forward():
            pass

    def __init__(self, latents: Optional[List['Model']]=None, output_dim: Optional[int]=None, input_dim: Optional[int]=None, W: Optional[np.ndarray] = None):
        print('CREATING')

        if input_dim is None:
            input_dim = len(latents)

        self.latents = objax.ModuleList(latents)
        self.output_dim = output_dim

        self.input_dim = input_dim

        #self.raw_W = objax.TrainVar(np.ones([self.output_dim, self.input_dim]))
        #self.raw_W = objax.TrainVar(np.eye(self.output_dim))
        self.raw_W = np.eye(self.output_dim)

        self.batches = objax.ModuleList([LMC.LMC_p(self.latents, self.raw_W[p]) for p in range(self.output_dim)])

        super(LMC, self).__init__()


    def get_batches(self):
        return self.batches

    def number_of_latents(self):
        return len(self.latents)


class GPRN(NonLinearTransform):
    r"""
    Gaussian Process Regression Network.

    Generative model:
        f_p = \sum w_{p,q}(x) g_q(x)
    """
    def __init__(self, latent_f: Optional[List['Model']]=None, latent_w: Optional[List[List['Model']]]=None, output_dim: Optional[int]=None, input_dim: Optional[int]=None):
        self.latent_f = latent_f
        self.latent_w = latent_w
        self.output_dim = output_dim
        self.input_dim = input_dim


class ConstrainedLMC(Transform):
    r"""
    Constrained Linear model of coregionilisation.

    Generative model:
        f_p = \sum w_{p,q} g_q
    """


class ConstrainedGPRN(Transform):
    r"""
    Constrained Gaussian Process Regression Network.

    Generative model:
        f_p = \sum w_{p,q}(x) g_q(x)
    """
