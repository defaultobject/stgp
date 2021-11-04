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
from ..batching import loop_or_batch

from . import Model, GP
from ..kernels import Kernel
from ..inference import Batch
from ..computation.log_marginal_likelihoods import * 
from ..likelihood import Gaussian
from ..kernels import RBF
from ..utils.utils import ensure_module_list
from ..transforms import Independent

from jax.tree_util import tree_structure
from jax.tree_util import tree_flatten, tree_unflatten, register_pytree_node

from jax.experimental import loops

from ..batching import Batched
from ..sparsity import NoSparsity


@obj_dispatch(Model, 'Batch')
class BatchGP(Model):
    def __init__(self, X=None, Y=None, inference: 'Batch'=None, likelihood: 'Likelihood'=None, kernel: 'Kernel'=None, prior: 'Transform' = None, latent=False, latent_y=False, **kwargs):

        super(BatchGP, self).__init__(**kwargs)

        self.X = X

        self.latent_y = latent_y
        if latent_y:
            #self.Y = objax.StateVar(np.array(Y))
            self.Y = objax.TrainVar(np.array(Y))
        else:
            self.Y = Y

        if kernel is not None:
            if type(kernel) is not list:
                kernel = [kernel]

        self.inference = inference
        self.likelihood = likelihood
        self.kernel = kernel
        self.prior = prior
        self.sparsity = NoSparsity(self.X) # Sparsity for batch GPs is not supported
        self.latent = latent

        self.num_latents = None
        self.num_outputs = None

        if not latent:
            # latent GPs are constructed internally to they will already have been setup properly
            self.setup_data()
            self.set_defaults()
            self.fix_inputs()

    def setup_data(self):
        """ Ensure input data is valid """

        if self.X is None:
            raise RuntimeError('X must be passed')

        self.X = np.array(self.X)

        # Input dimension / number of covariates or features
        self.D = self.X.shape[1]

        if (self.Y is not None) and (not self.latent_y):
            self.Y = np.array(self.Y)

    def set_defaults(self):
        """ Replace missing options with defaults """

        # Figure out which prior mode is being used (kernel vs prior)

        if (self.kernel is not None) and (self.prior is not None):
            raise RuntimeError('Only kernel or a prior must be passed')

        if self.prior is None:
            # construct an independent prior for each latent function

            if self.Y is not None:
                if self.latent_y:
                    self.num_outputs = self.Y.value.shape[1]
                else:
                    self.num_outputs = self.Y.shape[1]
            else:
                self.num_outputs = 1

            self.num_latents = self.num_outputs

            # Only set a default kernel if we are in kernel mode and one has not been passed
            if self.kernel is None:
                self.kernel = [
                    RBF(
                        lengthscales=[1.0 for d in range(self.D)],
                        input_dim=self.D
                    )
                    for j in range(self.num_latents)
                ]

            # Construct independent prior
            self.prior = Independent(
                latents = [
                    GP(
                        X = self.X,
                        kernel = self.kernel[q],
                        latent=True
                    )
                    for q in range(self.num_latents)
                ],
                prior=True
            )

        # Figure out how many latent functions are being used
        self.num_latents = self.prior.num_latents
        self.num_outputs = self.prior.num_outputs

        if self.inference == None:
            self.inference = Batch()

        if self.likelihood == None:
            self.likelihood = objax.ModuleList([Gaussian(variance=1.0) for j in range(self.num_outputs)])

    def fix_inputs(self):
        """ Convert all inputs into a consistent format """

        # We do not need to make kernel a module list because this is done within the prior object
        self.likelihood = ensure_module_list(self.likelihood)

    def get_objective(self, X=None, Y = None):
        if X is None:
            X, Y = self.X, self.Y

        nlml = self.inference.neg_log_marginal_likelihood(
            X,
            Y,
            self.likelihood,
            self.prior
        )

        chex.assert_rank(nlml, 0)

        return nlml

    def predictive_mu(self, XS):
        mu, _ = self.predict(XS, diagonal=True)
        return mu

    def predictive_covar(self, XS_1, XS_2):
        X = self.X

        if self.latent_y:
            Y = self.Y.value
        else:
            Y = self.Y

        var_arr =  self.inference.predictive_covar(
            XS_1, XS_2, X, Y, self.likelihood, self.prior
        )
        return var_arr

    def predict(self, XS, diagonal=True, squeeze=True):

        X = self.X

        if self.latent_y:
            Y = self.Y.value
        else:
            Y = self.Y

        mu_arr, var_arr =  self.inference.predict(
            XS, X, Y, self.likelihood, self.prior, diagonal=diagonal
        )

        if squeeze:
            mu_arr, var_arr = np.squeeze(mu_arr), np.squeeze(var_arr)

        return mu_arr, var_arr
