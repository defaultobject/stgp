import jax
from jax.scipy.optimize import minimize
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp

import objax
from objax import ModuleList, TrainRef, TrainVar
import numpy as np
from typing import List, Union

from .trainer import Trainer, vc_remove_vars
from ..utils.utils import vc_keep_vars, match_suffix

import legogp
from legogp.computation.natural_gradients.nat_grad import general_ell_natural_gradients



class NatGradTrainer(Trainer):
    def train(
        self, 
        model:'Model', 
        optimizer, 
        learning_rate, 
        epochs, 
        hold_vars = None,
        callback=None
    ):

        m = model
        vc = m.vars()

        m_name = match_suffix('._m', vc.keys())
        s_chol_name = match_suffix('._S_chol', vc.keys())
        approx_posterior_vars = [m_name, s_chol_name]

        vars_to_update = vc_keep_vars(vc, approx_posterior_vars)

        natgrad_fn = objax.Jit(
            model.natural_gradients,
            vc,
            static_argnums = (0, 1, 2,)
        )

        def gradient_step(i):
            params = natgrad_fn(
                learning_rate, approx_posterior_vars[0], approx_posterior_vars[1]
            )

            vars_to_update.assign(params)


        for i in range(epochs):
            gradient_step(i)


            if callback is not None:
                callback(i, None, None)

        return None, None

