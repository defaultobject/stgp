from . import Model
from .. import Parameter
from .. import Kernel
from .. import Likelihood
from .. import Inference
from .. import Sparsity

from ..inference import *
from ..computation import *
from ..likelihoods import GPRN_Likelihood

from ..computation.marginal_likelihoods import *
from ..computation.predictors import *
from ..decorators import return_gradients
from ..distributions import *

from . import GP
from . import VGPRN

import jax.numpy as np
from jax import jit, partial

import typing
from typing import Union, List, Optional


class GPRN(GP):
    def __init__(
        self,
        X: Union[np.ndarray, List[np.ndarray]],
        Y: Union[np.ndarray, List[np.ndarray]],
        kernel: Optional[Kernel] = None,
        likelihood: Optional[Likelihood] = None,
        inference: Optional[Inference] = None,
        sparsity: Optional[Sparsity] = None,
        options: Optional[dict] = None,
        name: Optional[str] = None,
        num_latents: Optional[int] = 1,
    ) -> None:
        self.save_inputs_to_properties(locals())

        super(GPRN, self).__init__(
            self.X,
            self.Y,
            self.kernel,
            self.likelihood,
            self.inference,
            self.sparsity,
            self.options,
            self.name,
        )

    def set_defaults_if_needed(self):
        # TODO(ollie): this should be done on the standarized inputs to avoid the if statement
        if type(self.Y) is list:
            self.num_outputs = len(self.Y)
        else:
            self.num_outputs = self.Y.shape[1]

        if self.likelihood is None:
            self.likelihood = GPRN_Likelihood(
                num_outputs=self.num_outputs, num_latents=self.num_latents
            )

        super(GPRN, self).set_defaults_if_needed()

    def get_model_dicts(self) -> dict:
        return {VariationalInference: VGPRN}

    def get_model(self):
        model_dicts = self.get_model_dicts()
        for key, func in model_dicts.items():
            if isinstance(self.inference, key):
                self.base_model = model_dicts[key](
                    self.X,
                    self.Y,
                    self.kernel,
                    self.likelihood,
                    self.inference,
                    self.sparsity,
                    self.options,
                    self.name,
                    self.num_latents,
                )
                return

        raise NotImplementedError(
            "GPRN with inference {inf} has not been implemented yet".format(
                inf=self.inference
            )
        )
