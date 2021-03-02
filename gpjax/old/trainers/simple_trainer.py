from . import Trainer
from .. import Model
from .. import Parameter
from .. import decorators

import jax
from jax.experimental import optimizers
from jax.test_util import check_grads
import jax.numpy as jnp

from jax.interpreters import xla

xla._xla_callable.cache_clear()

import time
import json

import numpy as np

import typing
from typing import Callable, Optional, List, Tuple

import warnings


from timeit import default_timer as timer


def get_dict_subset(d: dict, arr: list) -> dict:
    return {key: d[key] for key in arr}


def get_dict_keys_not_in_list(d: dict, arr: list) -> list:
    keys = []
    for k in d.keys():
        if k in arr:
            continue
        keys.append(k)
    return keys


class SimpleTrainer(Trainer):
    def __init__(
        self,
        model: Model,
        optimizer: Optional[Callable] = None,
        heartbeat_fn: Optional[Callable] = None,
        epochs: Optional[int] = None,
        hold_params: List[Tuple[List[str], str]] = None,
    ):
        """
        Args:
            hold_params: list of parameter names to hold during optimisation. When a list is passed inside hold_params this is used for hold many params. E.g
                    [['RBF', 'variances']]
                will hold all variables starting with RBF and have variances in the name

        """
        if epochs is None:
            epochs = 100
            warnings.warn("Epochs not specified. Using default number of 100.")

        if optimizer is None:
            optimizer = lambda: optimizers.adam(step_size=0.01)
            warnings.warn("Optimizer not specified. Adam with step size of 0.01.")

        if hold_params is None:
            hold_params = []

        self.epochs = epochs
        self.optimizer = optimizer
        self.hold_params = hold_params
        self.model = model

        self.opt_init = None
        self.opt_update = None
        self.get_params = None
        self.opt_state = None

    def train(self, epochs_counter=0) -> None:
        start = timer()
        if self.opt_init is None:
            opt_init, opt_update, get_params = self.optimizer()

            self.opt_init = opt_init
            self.opt_update = opt_update
            self.get_params = get_params

        # remove params in hold_params from the dictionary of trainable aprams
        params_to_remove = []
        for p in self.hold_params:
            for a in Parameter.PARAM_DICT.keys():
                if type(p) is list:
                    if False:
                        print(p)
                    if a.startswith(p[0]) and a.find(p[1]) != -1:
                        params_to_remove.append(a)

                elif a.startswith(p):
                    params_to_remove.append(a)

        param_names = get_dict_keys_not_in_list(Parameter.PARAM_DICT, params_to_remove)
        params = get_dict_subset(Parameter.PARAM_DICT, param_names)

        ls_name = param_names[0]

        if self.opt_state is None:
            self.opt_state = self.opt_init(params)

        elbos = []

        def gradient_step(i, opt_state):
            params = self.get_params(opt_state)

            epoch_step = epochs_counter + i

            start = time.process_time()
            elbo, gradients = self.model.get_objective(params=params, return_grad=True)

            time_taken = time.process_time() - start  # seconds
            elbos.append(elbo)

            if False:
                print(
                    "iter %2d: objective=%2.2f: %2.2f (s)"
                    % (epoch_step, elbo, time_taken)
                )

                def grad_print(grads):
                    for key, item in grads.items():
                        if len(item.shape) == 0 or np.sum(item.shape) > 5:
                            print("===== gradient ======= ", key, " -- ", np.sum(item))
                        else:
                            print("===== gradient ======= ", key, " -- ", item)

                if True:
                    grad_print(gradients)

            if np.isnan(elbo):
                raise RuntimeError("Nans")

            return self.opt_update(epoch_step, gradients, self.opt_state)
            # return self.opt_update(epoch_step, gradients, self.opt_state)
            # return self.opt_init(params)

        # elbo = self.model.get_objective(return_grad=False)
        # elbos.append(elbo)

        timer_step = timer()

        start = time.process_time()
        for i in range(self.epochs):
            start_epoch = time.process_time()
            self.opt_state = gradient_step(i, self.opt_state)

            epoch_time_taken = time.process_time() - start  # seconds

        time_taken = time.process_time() - start  # seconds

        timer_step_1 = timer()

        self.model.anchor(self.get_params(self.opt_state))

        timer_step_2 = timer()
        return elbos
