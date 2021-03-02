from . import Model, CVI_VGP
from .. import Parameter
from .. import Kernel
from .. import Likelihood
from .. import Inference
from .. import Sparsity

from ..data import Data, TimeseriesData

from . import SDE_GP
from ..inference import *
from ..distributions import *
from ..decorators import return_gradients

from ..likelihoods import DiagonalGaussianLikelihood

from ..settings import Settings

from ..approximate_posteriors import (
    GaussianApproxPosterior,
    DiagonalConjugateApproxPosterior,
)
from ..sparsity import NoSparsity

from .. import Dispatcher

import jax.numpy as np
from jax import jit, partial

import numpy as onp

import typing
from typing import Union, List, Optional

import warnings


@Dispatcher.register("gp_model", StateSpaceVI, None, 1)
class CVI_SDE_VGP(CVI_VGP):
    def setup_conjugate_model(self):

        self.data = TimeseriesData(self.X, self.Y, self.X_onp, self.Y_onp)

        # if type(self.sparsity) is not NoSparsity:
        #    raise NotImplementedError()

        if not self.inference.is_initialized():

            # construct a partial model to be setup on use
            self.batch_inference = StateSpace(name="StateSpace")

            approx_posterior = DiagonalConjugateApproxPosterior(
                data=self.data,
                sparsity=self.sparsity,
                conjugate_model=SDE_GP,
                batch_inference=batch_inference,
                kernel=self.kernel,
            )

            self.inference.initialize(approx_posterior)
