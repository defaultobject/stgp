from . import Trainer, SimpleTrainer
from .. import Model
from .. import Parameter

from .. import Dispatcher

from ..computation.natural_gradients import *

from .simple_trainer import get_dict_subset, get_dict_keys_not_in_list
import jax
from jax.experimental import optimizers
from jax.test_util import check_grads
import jax.numpy as jnp

import time

import numpy as np

import typing
from typing import Callable, Optional, List, Tuple

import warnings


def match_contains(arr, s):
    return [a for a in arr if s in a]


class NaturalGradients(SimpleTrainer):
    def __init__(
        self,
        model: Model,
        optimizer: Optional[Callable] = None,
        heartbeat_fn: Optional[Callable] = None,
        epochs: Optional[int] = None,
        hold_params: List[Tuple[List[str], str]] = None,
        step_size=None,
    ):
        super(NaturalGradients, self).__init__(
            model, optimizer, heartbeat_fn, epochs, hold_params
        )
        if step_size is None:
            step_size = 0.1
            warnings.warn(
                "step_size not specified. Using default number of {step_size}.".format(
                    step_size=step_size
                )
            )
        self.step_size = step_size

        self.nat_grad = Dispatcher.dispatch(
            "natural_gradients",
            type(self.model),
            type(self.model.inference),
            self.model.kernel.input_dim,
        )

    def train(self, epochs_counter=0) -> None:
        """
        Args:
            hold_params: list of parameter names to hold during optimisation. When a list is passed inside hold_params this is used for hold many params. E.g
                    [['RBF', 'variances']]
                will hold all variables starting with RBF and have variances in the name

        """
        # natural gradient state

        approximate_posterior_params = Parameter.SCOPE_DICT["variational"]
        params = get_dict_subset(Parameter.PARAM_DICT, approximate_posterior_params)

        params = Parameter.PARAM_DICT

        elbos = []

        def gradient_step(i, params):
            epoch_step = epochs_counter + i

            if False:
                print("# ", epoch_step, " ####################################")

            start = time.process_time()

            nat_params_new = self.nat_grad(self.model, self.step_size)

            if False:
                elbo = self.model.get_objective(return_grad=False)
                elbos.append(elbo)

            params.update(nat_params_new)

            time_taken = time.process_time() - start  # seconds
            if False:
                print("Time Taken: ", time_taken)

            if np.any(np.isnan(nat_params_new[list(nat_params_new.keys())[0]])):
                raise RuntimeError("Nans")

            return params

        start = time.process_time()
        for i in range(self.epochs):
            start = time.process_time()
            params = gradient_step(i, params)

            time_taken = time.process_time() - start  # seconds

        self.model.anchor(params)

        return elbos
