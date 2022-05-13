import jax
from scipy.optimize import minimize
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp

import objax
from objax import ModuleList, TrainRef, TrainVar
import numpy as np

from timeit import default_timer as timer

import json
import typing
from typing import List, Union

from ..utils.utils import vc_remove_vars, vc_keep_vars

class Trainer:
    """
    All trainers are initalised with:
        m: model object
        optimizer: [list[str], str]
        opt_args: [none, dict] 
        hold_vars: list of vars to not train

    In addition to hold_vars, if there any parameters that been held they will also not be trained

    This is so all the required functions can be jitted on initialisation, and then the trainer object
       can be reused without further jitting

    """

    def get_all_hold_vars(self, hold_vars):
        if hold_vars is None:
            hold_vars = []

        # Get variables that have train=False
        hold_vars += self.m.get_fixed_params()

        return hold_vars

    def __init__(self, m, optimizer, opt_args = None, hold_vars = None):
        if opt_args == None:
            opt_args = {}

        self.m = m

        # Collect variables to train
        all_vars = m.vars()

        hold_vars = self.get_all_hold_vars(hold_vars)

        if len(hold_vars) > 0:
            vars_to_train = vc_remove_vars(all_vars, hold_vars)
        else:
            vars_to_train = all_vars

        # Jit required functions
        objective_fn = objax.Jit(self.m.get_objective, all_vars)

        self.grad_fn = objax.Jit(
            objax.GradValues(objective_fn, vars_to_train), 
            all_vars
        )

        self.objective_fn = objective_fn

        # Get optimizer
        self.opt = optimizer(vars_to_train, **opt_args)
        self.vars_to_train = vars_to_train
        self.all_vars = all_vars

class ScipyTrainer(Trainer):
    """
    A simple wrapper around scipy optimizers.

    Example:

    learning_curve, training_time = ScipyTrainer().train(
        m, 
        'BFGS',
        0.01,
        epochs,
        callback = None

    Heavily based on https://gist.github.com/slinderman/24552af1bdbb6cb033bfea9b2dc4ecfd with modifications to work with objax
    """
    def __init__(self, m, optimizer, opt_args = None, hold_vars = None):

        if opt_args == None:
            opt_args = {}

        self.m = m
        self.optimizer = optimizer

        # Collect variables to train
        all_vars = self.m.vars()

        hold_vars = self.get_all_hold_vars(hold_vars)

        m_vc = m.vars()

        # Only keep the trainable vars without the hold vars
        trainable_vc = m_vc.subset(TrainVar)

        if len(hold_vars) > 0:
            trainable_vc = vc_remove_vars(trainable_vc, hold_vars)
        else:
            trainable_vc = trainable_vc

        self.trainable_vc = trainable_vc

        # Jit required functions
        objective_fn = objax.Jit(self.m.get_objective, all_vars)

        self.grad_fn = objax.Jit(
            objax.Grad(objective_fn, self.trainable_vc), 
            all_vars
        )

        self.objective_fn = objective_fn

    def train(self, learning_rate, epochs, callback=None):
        """ For consistency we accept learning_rate here although it is not used. """

        x0 = self.trainable_vc.tensors()
        x0_flat, unravel = ravel_pytree(x0)

        def fun_flat(x_flat):
            self.trainable_vc.assign(unravel(x_flat))
            return self.objective_fn()

        def grad_flat(x_flat):
            # Convert from flat to pytree and assign
            self.trainable_vc.assign(unravel(x_flat))

            # evaluate gradient
            g_flat, _ = ravel_pytree(self.grad_fn())

            return np.array(g_flat)

        learning_rates = []

        # Wrap the callback to consume a pytree
        def callback_wrapper(x_flat, *args):
            learning_rates.append(fun_flat(x_flat))

            if callback is not None:
                callback(None, None, None)

        results = minimize(
            fun_flat, 
            x0_flat, 
            method=self.optimizer, 
            jac = grad_flat,
            callback = callback_wrapper,
            options = {
                'disp': False,
                'maxiter': epochs
            }
        )

        res_x = unravel(results.x)
        self.trainable_vc.assign(res_x)

        return jnp.array(learning_rates).flatten(), 0


class GradDescentTrainer(Trainer):
    def train(
        self,
        learning_rate,
        epochs,
        callback=None,
        epoch_ofset = None
    ):
        start = timer()
        epoch_arr = []

        def train_op():
            grad, val = self.grad_fn()
            self.opt(learning_rate, grad)
            return grad, val

        for i in range(epochs):
            grad, val = train_op()


            if np.isnan(val):
                print(grad)
                raise RuntimeError('NaN encountered whilst training!')

            if callback is not None:
                callback(i, grad, val)

            # Clean up val
            epoch_arr.append(jnp.array(val).flatten())

        end = timer()
        training_time = end - start

        return jnp.array(epoch_arr).flatten(), training_time

class SwitchTrainer(Trainer):
    """
    For use when multiple trainers are used per training epoch.

    Example:

        # Only train approximate posterior through natural gradients
        for q in m.approximate_posterior.approx_posteriors:
            q._m.fix()
            q._S_chol.fix()

        grad_step = GradDescentTrainer(m, objax.optimizer.Adam)
        nat_grad_step = NatGradTrainer(m)

        trainer = SwitchTrainer(
            [grad_step, nat_grad_step],

        )
        trainer.train(
            100,
            [0.01, 1.0],
            [1, 1],
            None
        )

    We pass through the trainers grad_step, and nat_grad_step through the init function to minimize jitting.

    """
    def __init__(self, trainer_list: list):
        self.trainer_list = trainer_list

    def train(
        self,
        learning_rates: list,
        epochs: list,
        callback = None

    ):
        iters = epochs[1]
        epochs = int(epochs[0])

        start = timer()

        num_trainers = len(self.trainer_list)

        total_elbos = []
        completed_epochs = [0 for j in range(num_trainers)]

        for i in range(epochs):
            for j in range(num_trainers):
                lc_j, _ = self.trainer_list[j].train(
                    learning_rates[j], 
                    iters[j], 
                    None, # We do not support individual trainer callbacks
                    epoch_ofset = completed_epochs[j]
                )

                total_elbos.append(lc_j)

                completed_epochs[j] += iters[j]

            # After calling all individual trainers we have completed one training epoch
            if callback is not None:
                callback(i, None, None)

        end = timer()
        training_time = end - start

        return total_elbos, training_time
