from . import Inference

import typing


class VariationalInference(Inference):
    def __init__(self, variational_posterior=None, name=None) -> None:
        self.initialized = not (variational_posterior == None)
        self.variational_posterior = variational_posterior

        self.name = name or self.__class__.__name__

        super(VariationalInference, self).__init__(self.name)

    def initialize(self, approx_posterior: "ApproximatePosterior") -> None:
        self.initialized = True
        self.variational_posterior = approx_posterior

    def is_initialized(self) -> bool:
        return self.initialized
