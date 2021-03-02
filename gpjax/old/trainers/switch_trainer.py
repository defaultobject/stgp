from . import Trainer
from .. import Model
from .. import Parameter

import jax
from jax.experimental import optimizers
from jax.test_util import check_grads
import jax.numpy as jnp

import time
import json

import numpy as np

import typing
from typing import Callable, Optional, List, Tuple

import warnings


class SwitchTrainer(Trainer):
    def __init__(
        self,
        trainer_arr: List[Trainer],
        heartbeat_fn: Optional[Callable] = None,
        epochs: Optional[int] = None,
    ):
        self.trainer_arr = trainer_arr

        if epochs is None:
            epochs = 100
            warnings.warn(
                "Epochs not specified. Using default number of {epochs}.".format(
                    epochs=epochs
                )
            )

        self.epochs = epochs

        self.heartbeat_fn = heartbeat_fn

    def train(self):
        start = time.process_time()

        total_elbos = []
        total_epochs = 0

        # stores the epochs for each of the subtrainers
        epochs_counter = [0 for j in range(len(self.trainer_arr))]

        train_flag = True
        for i in range(self.epochs):
            j = 0
            for trainer in self.trainer_arr:
                if False:
                    if j == 1:
                        # need to adjust the epoch so the step size schedule is correct
                        _epoch = int((total_epochs - 1) / 2)
                    else:
                        _epoch = int((total_epochs) / 2)

                elbos = trainer.train(epochs_counter[j])
                total_elbos += elbos

                epochs_counter[j] += trainer.epochs
                total_epochs += trainer.epochs

                if self.heartbeat_fn is not None:
                    continue_flag = self.heartbeat_fn(total_epochs)
                    if continue_flag == False:
                        # break out of training
                        print("breaking training")
                        train_flag = False
                        break

                j += 1
            if train_flag == False:
                print("training end")
                break

        time_taken = time.process_time() - start  # seconds
        return np.array(total_elbos)
