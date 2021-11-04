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
from ..kernels import Kernel
from ..inference import Variational
from ..transforms import Transform, Identity
from ..computation.log_marginal_likelihoods import *
from ..likelihood import Gaussian
from ..kernels import RBF
from ..approximate_posteriors import MeanFieldApproximatePosterior
from ..sparsity import NoSparsity
from ..utils.utils import ensure_module_list


@obj_dispatch(Model, 'Variational', 'NoSparsity')
class VGP(Model):
    def __init__(self, X=None, Y=None, inference: 'Variational'=None, likelihood: 'Likelihood'=None, kernel=None, prior: 'Transform'=None, sparsity=None, whiten=False, minibatch=None, **kwargs):
        super(VGP, self).__init__(**kwargs)

        self.X = X
        self.Y = Y

        self.inference = inference
        self.likelihood = likelihood
        self.prior = prior
        self.kernel = kernel
        self.approximate_posterior = None
        self.sparsity = sparsity
        self.whiten = whiten
        self.minibatch = minibatch

        self.set_defaults()

    def predict(self, XS, diagonal=True, squeeze=True):

        mean, var = self.inference.predict(
            XS, 
            self.X, 
            self.likelihood, 
            self.prior,
            self.approximate_posterior,
            diagonal=diagonal
        )

        if squeeze:
            return np.squeeze(mean), np.squeeze(var)

        return mean, var

    def setup_data(self):
        """ Ensure input data is valid """

        if self.X is None:
            raise RuntimeError('X must be passed')

        if self.Y is None:
            raise RuntimeError('Y must be passed')

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

    def fix_inputs(self):
        """ Convert all inputs into a consistent format """

        self.kernel = ensure_module_list(self.kernel)
        self.likelihood = ensure_module_list(self.likelihood)
        self.sparsity = ensure_module_list(self.sparsity)

    def set_defaults(self):
        # Input dimension / number of covariates or features
        self.D = self.X.shape[1]

        self.num_latents = self.Y.shape[1]

        if self.prior == None:
            self.prior = Identity()

        if self.sparsity == None:
            self.Nq = self.X.shape[0]
        else:
            self.Nq = self.sparsity[0].Z.shape[0]


        if self.sparsity == None:
            #X is treated as Z
            self.sparsity = objax.ModuleList([NoSparsity(self.X) for j in range(self.num_latents)])

        if self.inference == None:
            self.inference = Variational()


        if False:
            if self.approximate_posterior is None:
                self.approximate_posterior = MeanFieldApproximatePosterior(self.prior, whiten=self.whiten)

        if self.kernel == None:
            self.kernel = objax.ModuleList([
                RBF(
                    lengthscales=[1.0 for d in range(self.D)],
                    input_dim=self.D
                )
                for j in range(self.num_latents)
            ])

        if self.likelihood == None:
            self.likelihood = objax.ModuleList([Gaussian(variance=1.0) for j in range(self.num_latents)])

    def get_objective(self):

        return 0.0

        elbo = self.inference.ELBO(
            self.X,
            self.Y,
            self.likelihood,
            self.prior,
            self.approximate_posterior,
            self.minibatch
        )

        return -elbo

