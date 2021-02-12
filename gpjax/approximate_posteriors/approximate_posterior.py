from .. import Likelihood
from .. import Distribution
from ..module import Module

import abc
from abc import ABC
from abc import abstractmethod

import jax
import jax.numpy as np

import typing
from typing import Optional, List

class ApproximatePosterior(ABC, Module):
    def __init__(self, name:Optional[str]=None ):
        self.name = name
        super(ApproximatePosterior, self).__init__(name=name)

    @abstractmethod 
    def KL(self, distribution: Distribution) -> np.ndarray:
        pass

    @abstractmethod 
    def expected_log_likelihood(self, X:np.ndarray, Y:np.ndarray, likelihood: Likelihood) -> np.ndarray:
        pass

