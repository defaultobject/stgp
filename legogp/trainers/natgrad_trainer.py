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
    def __init__(
        self, 
        model,
        hold_vars = None,
        schedule='constant',
        total_epochs=None
    ):
        self.m = model
        vc = self.m.vars()

        m_name = match_suffix('._m', vc.keys())
        s_chol_name = match_suffix('._S_chol', vc.keys())
        self.approx_posterior_vars = [m_name, s_chol_name]

        self.vars_to_update = vc_keep_vars(vc, self.approx_posterior_vars)

        self.natgrad_fn = objax.Jit(
            self.m.natural_gradients,
            vc,
            static_argnums = (1, 2,)
        )

        self.objective_fn = objax.Jit(self.m.get_objective, vc)

        self.schedule = schedule
        self.total_epochs = total_epochs



    def train(
        self, 
        learning_rate, 
        epochs, 
        callback=None,
        epoch_ofset=0
    ):

        def gradient_step(i, global_i):
            if self.total_epochs:
                percent = (global_i+1)/self.total_epochs
            else:
                percent = (i+1)/epochs
            if self.schedule == 'linear':
                lr = learning_rate[1] * percent + (1-percent) * learning_rate[0]
            if self.schedule == 'log':
                lr = np.power(learning_rate[1], percent) * np.power(learning_rate[0], (1-percent))
            elif self.schedule == 'constant':
                lr = learning_rate
            else:
                raise NotImplementedError(f'{scheudle} is not implemented')

            print(f'{i} / {epochs} -- {global_i} / {self.total_epochs} -- {lr}')

            params = self.natgrad_fn(
                lr, self.approx_posterior_vars[0], self.approx_posterior_vars[1]
            )

            if np.any(np.isnan(params[0])):
                raise RuntimeError('NaN encountered whilst natgrad training!')

            self.vars_to_update.assign(params)

        epoch_arr = []
        for i in range(epochs):
            val = self.objective_fn()

            epoch_arr.append(val)

            gradient_step(i, i+epoch_ofset)

            if callback is not None:
                callback(i, None, None)

        return epoch_arr, None

