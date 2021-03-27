import objax
import jax
import jax.numpy as np
import numpy as onp
from typing import Optional, Tuple
import chex

from .. import settings
from ..decorators import strict_mode_check, ensure_data
from ..obj_dispatch import obj_dispatch, obj_find
from ..dispatch import evoke

from . import Model
from ..kernel import Kernel
from ..inference import Variational
from ..transform import Transform
from ..computation.log_marginal_likelihoods import *
from ..likelihood import Gaussian
from ..kernel import RBF
from ..approximate_posteriors import GaussianApproximatePosterior
from ..sparsity import NoSparsity

from jax.tree_util import tree_structure
from jax.tree_util import tree_flatten, tree_unflatten, register_pytree_node

from jax.experimental import loops



@obj_dispatch(Model, 'Variational', 'NoSparsity')
class VGP(Model):
    def __init__(self, X=None, Y=None, inference: 'Variational'=None, likelihood: 'Likelihood'=None, kernel: 'Kernel'=None, whiten=False):
        super(VGP, self).__init__()
        self.X = X
        self.Y = Y
        self.inference = inference
        self.likelihood = likelihood
        self.kernel = kernel
        self.approximate_posterior = None
        self.sparsity = None
        self.whiten = whiten

        self.set_defaults()

    def predict(self, XS, diagonal=True, squeeze=True):
        if diagonal is False:
            raise NotImplementedError()

        mean, var = self.inference.predict(
            XS, 
            self.X, 
            self.likelihood, 
            self.kernel,
            self.sparsity,
            self.approximate_posterior
        )

        if squeeze:
            return np.squeeze(mean), np.squeeze(var)

        return mean, var

    def set_defaults(self):
        self.num_latents = self.Y.shape[1]

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

        self.Nq = self.X.shape[0]
        self.D = self.X.shape[1]

        self.dim  = self.num_latents*self.Nq    

        if self.sparsity == None:
            #X is treated as Z
            self.sparsity = [NoSparsity(self.X) for j in range(self.num_latents)]

        if self.inference == None:
            self.inference = Variational()


        if self.approximate_posterior is None:
            self.approximate_posterior = objax.ModuleList([
                GaussianApproximatePosterior(self.Nq, whiten=self.whiten) for j in range(self.num_latents)
            ])

        self.approximate_posterior.whiten = self.whiten

        if self.kernel == None:
            self.kernel = objax.ModuleList([RBF(input_dim=self.D) for j in range(self.num_latents)])

        if self.likelihood == None:
            self.likelihood = objax.ModuleList([Gaussian() for j in range(self.num_latents)])
        else:
            if type(self.likelihood) is not list:
                self.likelihood = [self.likelihood]

            self.likelihood = objax.ModuleList(self.likelihood)

    def get_objective(self):
        elbo = self.inference.ELBO(
            self.X,
            self.Y,
            self.likelihood,
            self.kernel,
            self.sparsity,
            self.approximate_posterior,
        )

        return -elbo

