"""Multi-output/task specific transforms."""
from .transform import Transform
import typing
from typing import List, Optional
import jax.numpy as np


class LMC(Transform):
    r"""
    Linear model of coregionilisation.

    Generative model:
        f_p = \sum w_{p,q} g_q
    """
    def __init__(self, latents: Optional[List['Model']]=None, output_dim: Optional[int]=None, input_dim: Optional[int]=None, W: Optional[np.ndarray] = None):
        self.latents = latents
        self.output_dim = output_dim
        self.input_dim = input_dim


class GPRN(Transform):
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
