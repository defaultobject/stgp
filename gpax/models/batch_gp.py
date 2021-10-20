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

from . import Model
from ..kernel import Kernel
from ..inference import Batch
from ..computation.log_marginal_likelihoods import * 
from ..likelihood import Gaussian
from ..kernel import RBF

from jax.tree_util import tree_structure
from jax.tree_util import tree_flatten, tree_unflatten, register_pytree_node

from jax.experimental import loops

from ..batching import Batched
from ..sparsity import NoSparsity


@obj_dispatch(Model, 'Batch')
class BatchGP(Model):
    def __init__(self, X=None, Y=None, inference: 'Batch'=None, likelihood: 'Likelihood'=None, kernel: 'Kernel'=None, **kwargs):

        super(BatchGP, self).__init__(**kwargs)

        self.X = X
        self.Y = Y
        self.inference = inference
        self.likelihood = likelihood
        self.kernel = kernel
        self.num_latents = 1
        self.sparsity = NoSparsity(self.X)

        self.setup_data()
        self.set_defaults()
        self.fix_inputs()

    def setup_data(self):
        """ Ensure input data is valid """

        if self.X is None:
            raise RuntimeError('X must be passed')

        self.X = np.array(self.X)

        if self.Y is not None:
            self.Y = np.array(self.Y)


    def set_defaults(self):
        """ Replace missing options with defaults """

        # Input dimension / number of covariates or features
        self.D = self.X.shape[1]

        if self.Y is not None:
            self.num_latents = self.Y.shape[1]

        if self.inference == None:
            self.inference = Batch()

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

    def fix_inputs(self):
        """ Convert all inputs into a consistent format """

        if type(self.kernel) is not objax.ModuleList:
            if type(self.kernel) is not list:
                self.kernel = [self.kernel]

            self.kernel = objax.ModuleList(self.kernel)



        if type(self.likelihood) is not objax.ModuleList:
            if type(self.likelihood) is not list:
                self.likelihood = [self.likelihood]

            self.likelihood = objax.ModuleList(self.likelihood)

    def get_objective(self):
        lml_fn = evoke('log_marginal_likelihood')

        def _lml(X, Y, lik,  kernel):
            N = X.shape[0]
            Y = Y[:, None]

            return lml_fn(self.X, Y, lik, kernel)

        lml_arr = loop_or_batch(
            _lml,
            [ self.X, self.Y, self.likelihood, self.kernel ],
            [ None, 1, 0, 0 ],
            self.num_latents,
            num_outputs=1
        )

        nlml = - np.sum(lml_arr)

        chex.assert_rank(nlml, 0)

        return nlml


    def predict(self, XS, diagonal=True, squeeze=True):

        if diagonal:
            pred_fn = evoke('predict_diagonal')
        else:
            pred_fn = evoke('predict')

        def _predict(XS, X, Y, lik,  kernel):
            Y = Y[:, None]
            return pred_fn(XS, X, Y, lik, kernel)

        mu_arr, var_arr = loop_or_batch(
            _predict,
            [XS, self.X, self.Y, self.likelihood, self.kernel],
            [None, None, 1, 0, 0],
            self.num_latents,
            num_outputs=2
        )

        if squeeze:
            mu_arr, var_arr = np.squeeze(mu_arr), np.squeeze(var_arr)

        return mu_arr, var_arr
