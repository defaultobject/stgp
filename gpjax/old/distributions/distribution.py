from ..likelihoods import Likelihood
from ..module import Module

from abc import ABC
from abc import abstractmethod

import typing


class Distribution(Module):
    def __init__(self, name: str, meta: dict, trainable: bool) -> None:
        super(Distribution, self).__init__(name)

        self.name = name
        self.meta = meta

    @abstractmethod
    def log_likelihood_expectation(self, likelihood):
        # perform quadrature or monte carlo
        pass

    @abstractmethod
    def KL(self, distribution):
        # perform KL
        pass
