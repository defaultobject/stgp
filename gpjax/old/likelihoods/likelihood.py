from ..module import Module

from abc import ABC
from abc import abstractmethod

import typing


class Likelihood(Module):
    def __init__(self, name: str, meta: dict, trainable: bool) -> None:
        super(Likelihood, self).__init__(name)

        self.name = name
        self.meta = meta
        self.trainable = trainable

    @abstractmethod
    def moment_match(self, y, mu, var, hyp):
        pass

    def predict_requires_approximation(self):
        return not hasattr(self, "predict_mean_var")
