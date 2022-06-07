from .transform import Transform, LinearTransform, NonLinearTransform, Independent
import jax.numpy as np


class Aggregate(LinearTransform):
    def __init__(self, latents):

        super(Transform, self).__init__()

        self._latent_obj = latents



    def forward(self, f):
        raise NotImplementedError()


