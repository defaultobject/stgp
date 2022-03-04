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
from ..utils.utils import vc_keep_vars, match_suffix, get_parameters, get_var_name_with_id

from ..dispatch import _ensure_str

def get_vars_to_update(model, vc):
    approx_posterior = model.approximate_posterior

    param_dict = get_parameters(model, replace_name=False, return_id=True)

    if _ensure_str(approx_posterior) == 'MeanFieldApproximatePosterior':
        m_name_list = []
        S_chol_name_list = []
        for q in approx_posterior.approx_posteriors:
            m_name = get_var_name_with_id(model, id(q._m.raw_var), param_dict)
            S_chol_name = get_var_name_with_id(model, id(q._S_chol.raw_var), param_dict)
            m_name_list.append(m_name)
            S_chol_name_list.append(S_chol_name)
            
    elif _ensure_str(approx_posterior) == 'FullGaussianApproximatePosterior':
        pass
    else:
        raise RuntimeError()

    return vc_keep_vars(vc, [*m_name_list, *S_chol_name_list])

def update_vars(model, vars_to_update, params):
    approx_posterior = model.approximate_posterior

    if _ensure_str(approx_posterior) == 'MeanFieldApproximatePosterior':
        q_arr = approx_posterior.approx_posteriors

        new_params = []
        for q in range(len(q_arr)):
            new_params += [params[0][q], params[1][q]]

        vars_to_update.assign(new_params)
    else:
        raise RuntimeError()


class NatGradTrainer(Trainer):
    def __init__(
        self, 
        model,
        hold_vars = None,
        schedule=None,
        total_epochs=None
    ):
        self.m = model
        vc = self.m.vars()

        self.vars_to_update = get_vars_to_update(self.m, vc)

        self.natgrad_fn = objax.Jit(
            self.m.natural_gradients,
            vc
        )

        self.objective_fn = objax.Jit(self.m.get_objective, vc)

        if schedule is None:
            schedule = 'constant'

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

            elif self.schedule == 'log':
                lr = np.power(learning_rate[1], percent) * np.power(learning_rate[0], (1-percent))

            elif self.schedule == 'constant':
                lr = learning_rate
            else:
                raise NotImplementedError(f'{self.schedule} is not implemented')

            print(f'{i} / {epochs} -- {global_i} / {self.total_epochs} -- {lr}')

            params = self.natgrad_fn(lr)

            if np.any(np.isnan(params[0])):
                raise RuntimeError('NaN encountered whilst natgrad training!')

            update_vars(self.m, self.vars_to_update, params)

        epoch_arr = []
        for i in range(epochs):
            val = self.objective_fn()

            epoch_arr.append(val)

            gradient_step(i, i+epoch_ofset)

            if callback is not None:
                callback(i, None, None)

        return epoch_arr, None

