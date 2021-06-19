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
from ..inference import Batch
from ..transform import Transform
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
    def __init__(self, X=None, Y=None, inference: 'Batch'=None, likelihood: 'Likelihood'=None, kernel: 'Kernel'=None):
        super(BatchGP, self).__init__()
        self.X = X
        self.Y = Y
        self.inference = inference
        self.likelihood = likelihood
        self.kernel = kernel
        self.num_latents = 1
        self.sparsity = NoSparsity(self.X)

        self.set_defaults()

    def set_defaults(self):
        if self.Y is not None:
            self.num_latents = self.Y.shape[1]
            self.Y = np.array(self.Y)

        self.X = np.array(self.X)
        self.D = self.X.shape[1]


        if self.inference == None:
            self.inference = Batch()

        if self.kernel == None:
            self.kernel = objax.ModuleList([RBF(lengthscales=[1.0 for d in range(self.D)]) for j in range(self.num_latents)])

        elif type(self.kernel) is not list:
            self.kernel = objax.ModuleList([self.kernel])

        if self.likelihood == None:
            self.likelihood = objax.ModuleList([Gaussian(variance=1.0) for j in range(self.num_latents)])
        elif type(self.likelihood) is not list:
            self.likelihood = objax.ModuleList([self.likelihood])

    def get_objective(self):
        lml_fn = evoke('log_marginal_likelihood')


        if settings.use_loop_mode:

            lml = 0
            for q in range(self.num_latents):
                lml_q = lml_fn(self.X, self.Y[:, q][:, None], self.likelihood[q], self.kernel[q])
                lml += lml_q

            nlml = -lml

        else:
            with Batched(self.likelihood) as likelihood, Batched(self.kernel) as kernel:

                def lml(X, Y, lik, lik_vars, kernel, kernel_vars):
                    N = X.shape[0]
                    Y = Y[:, None]

                    lik.set_vars(lik_vars)
                    kernel.set_vars(kernel_vars)

                    return lml_fn(self.X, Y, lik.get_obj(), kernel.get_obj())

                lml = jax.vmap(lml, (None, 1, None, 0, None, 0))(self.X, self.Y, likelihood, likelihood.get_vars(), kernel, kernel.get_vars())

            nlml = -np.sum(lml)

        chex.assert_rank(nlml, 0)

        return nlml


    def predict(self, XS, diagonal=True, squeeze=True):

        if diagonal:
            pred_fn = evoke('predict_diagonal')
        else:
            pred_fn = evoke('predict')

        mean, var =pred_fn(
            XS, 
            self.X, 
            self.Y,
            self.likelihood[0], 
            self.kernel[0],
        )

        if squeeze:
            return np.squeeze(mean), np.squeeze(var)

        return mean, var
