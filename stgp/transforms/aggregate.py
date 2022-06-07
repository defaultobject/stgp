from .transform import Transform, LinearTransform, NonLinearTransform, Independent
import jax.numpy as np


class Aggregate(LinearTransform):
    def __init__(self, latents):

        super(Transform, self).__init__()

        # Allow passing a list of prior models and transformed model
        if type(latents) is list:
            self._latent_obj = Independent(latents=latents, prior=True)
        else:
            self._latent_obj = latents

    @property
    def latents(self):
        return self.latent_obj._latents_arr

    def forward(self, f):
        raise NotImplementedError()
