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


class SimpleTrainer(Trainer):
    def summary(self, train_vars):
        print(train_vars)

    def train(self, m, optimizer, learning_rate, epochs, callback=None):
        train_vars = m.vars()

        self.summary(train_vars)

        objective_fn = objax.Jit(m.get_objective, train_vars)
        grad_fn = objax.Jit(objax.GradValues(objective_fn, train_vars), train_vars)

        opt = optimizer(train_vars)

        start = timer()

        epoch_arr = []

        def train_op():
            grad, val = grad_fn()
            opt(learning_rate, grad)
            return grad, val

        for i in range(epochs):
            grad, val = train_op()

            if np.isnan(val):
                raise RuntimeError('NaN encountered whilst training!')

            if callback is not None:
                callback(i, grad, val)

            epoch_arr.append(val)

        end = timer()
        training_time = end - start

        return epoch_arr, training_time
