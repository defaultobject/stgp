import jax
from jax.scipy.optimize import minimize
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
    pass

class ScipyTrainer(Trainer):
    def train(self, m, optimizer, learning_rate, epochs, callback=None):
        vc = m.vars()

        objective_fn = objax.Jit(m.get_objective, vc)

        train_vars = ModuleList(TrainRef(x) for x in vc.subset(TrainVar))

        x0 = vc.tensors()
        x0_flat, unravel = ravel_pytree(x0)

        def fun_flat(x0_flat):
            vc.assign(unravel(x0_flat))
            return objective_fn()

        results = minimize(fun_flat, x0_flat, method='BFGS')

        res_x = unravel(results.x)
        vc.assign(res_x)

        return [], 0


class GradDescentTrainer(Trainer):
    def __init__(
        self, 
        models: Union['Model', List['Model']], 
        optimizer, 
        hold_vars = None,
        keep_vars = None,
    ):
        if type(models) is not list:
            models = [models]

        self.models = models

        train_vars = models[0].vars()

        def objective():
            obj = 0.0
            for m in models:
                obj += m.get_objective()
            return obj

        objective_fn = objax.Jit(objective, train_vars)

        if hold_vars is not None:
            vars_to_train = vc_remove_vars(train_vars, hold_vars)
        elif keep_vars is not None:
            vars_to_train = vc_keep_vars(train_vars, keep_vars)
        else:
            vars_to_train = train_vars

        self.grad_fn = objax.Jit(
            objax.GradValues(objective_fn, vars_to_train), 
            train_vars
        )

        aa = models[0].vars()

        seen = set()

        for v in aa.values():
            if id(v) not in seen:
                seen.add(id(v))
            else:
                #print(v)
                pass

        self.opt = optimizer(vars_to_train)

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

            epoch_arr.append(val)

        end = timer()
        training_time = end - start

        return epoch_arr, training_time


class SimpleTrainer(Trainer):
    def summary(self, train_vars):
        print(train_vars)

    def train(
        self, 
        models: Union['Model', List['Model']], 
        optimizer, 
        learning_rate, 
        epochs, 
        hold_vars = None,
        keep_vars = None,
        callback=None
    ):

        if hold_vars is None:
            hold_vars = []

        if type(models) is not list:
            models = [models]

        #models = objax.ModuleList(models)

        # Assume that models[0] is the 'global' model
        train_vars = models[0].vars()

        def objective():
            obj = 0.0
            for m in models:
                obj += m.get_objective()
            return obj

        objective_fn = objax.Jit(objective, train_vars)

        hold_vars += models[0].get_fixed_params()

        if len(hold_vars) > 0:
            vars_to_train = vc_remove_vars(train_vars, hold_vars)
        elif keep_vars is not None:
            vars_to_train = vc_keep_vars(train_vars, keep_vars)
        else:
            vars_to_train = train_vars

        grad_fn = objax.Jit(objax.GradValues(objective_fn, vars_to_train), train_vars)
        opt = optimizer(vars_to_train)

        start = timer()

        epoch_arr = []

        def train_op():
            grad, val = grad_fn()
            opt(learning_rate, grad)
            return grad, val

        for i in range(epochs):
            grad, val = train_op()

            if np.isnan(val):
                print(grad)
                raise RuntimeError('NaN encountered whilst training!')

            if callback is not None:
                callback(i, grad, val)

            epoch_arr.append(val)

        end = timer()
        training_time = end - start

        return epoch_arr, training_time


class SwitchTrainer(Trainer):
    def __init__(
        self,
        trainer_list,
        iters,
        learning_rate_list,
        epoch_list,
        callback_list = None
    ):
        self.trainer_list = trainer_list
        self.iters = iters
        self.learning_rate_list = learning_rate_list
        self.epoch_list = epoch_list

        if callback_list is None:
            callback_list = [None] * len(trainer_list)

        self.callback_list = callback_list

    def train(self):
        start = timer()

        num_trainers = len(self.trainer_list)

        total_elbos = []
        epochs = [0 for j in range(num_trainers)]

        for i in range(self.iters):
            for j in range(num_trainers):
                epochs_j, _ = self.trainer_list[j].train(
                    self.learning_rate_list[j], 
                    self.epoch_list[j], 
                    self.callback_list[j],
                    epoch_ofset = epochs[j]
                )

                total_elbos.append(epochs_j)
                epochs[j] += self.epoch_list[j]

        end = timer()
        training_time = end - start

        return total_elbos, training_time
