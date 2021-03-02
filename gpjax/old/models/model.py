from .. import Parameter
from .. import Kernel
from .. import Likelihood
from .. import Sparsity

from ..kernels import RBF
from ..likelihoods import GaussianLikelihood
from ..inference import BatchInference
from ..module import Module
from ..sparsity import NoSparsity

import typing
from typing import Union, List, Optional

import jax.numpy as np
import jax

import numpy as onp

from abc import ABC
from abc import abstractmethod

import warnings

import json
import pickle

from ..settings import Settings

is_list = lambda a: type(a) is list
check_shape_last_dim = lambda A, shp: np.all([a.shape[-1] == shp for a in A])
copy_to_list = lambda a, P: [a.copy() for p in range(P)]


class Model(ABC, Module):
    """
    Base class for models.
    """

    def __init__(
        self,
        X: Union[onp.ndarray, np.ndarray, List[np.ndarray]] = None,
        Y: Optional[Union[onp.ndarray, np.ndarray, List[np.ndarray]]] = None,
        kernel: Optional[Kernel] = None,
        likelihood: Optional[Likelihood] = None,
        inference: Optional["Inference"] = None,
        sparsity: Optional[Sparsity] = None,
        options: Optional[dict] = None,
        name: Optional[str] = None,
        X_onp: Optional[onp.ndarray] = None,
        Y_onp: Optional[onp.ndarray] = None,
        set_defaults: Optional[bool] = True,
        key=None,
    ) -> None:
        """
        X: an array or list of observation inputs - [NxD]
        Y: an array or list of observation outputs - [Nx1]
        kernel: kernel that defines GP prior. If not passed default RBF will be used.
        likelihood: likelihood that defines the measurement model. If not defined Gaussian will be used.
        inference: infeence object that defines the inference scheme. If not defined exact will be used..
        """

        # Not all models use X_onp, so have to manually save here
        set_defaults = True
        if key is None:
            key = jax.random.PRNGKey(Settings.seed)

        self.X_onp = X_onp
        self.Y_onp = Y_onp

        # sets all arguments of __init__ to be properties of this object
        self.save_inputs_to_properties(locals())

        self.standarize_x()

        # this is useful for constructing a 'partial' model that will be
        # initialised later
        if set_defaults:
            self.set_defaults_if_needed()

            self.standarize_inputs()
            self.check_model()

        if self.X_onp is None:
            self.X_onp = self.X.copy()

        if self.Y_onp is None and self.Y is not None:
            self.Y_onp = self.Y.copy()

        self.setup()

        super(Model, self).__init__(name=self.name)

    def standarize_x(self):
        # TODO: generalise and cleanup
        self.X_arr = [self.X] if not is_list(self.X) else self.X
        self.num_outputs = len(self.X_arr)

    def standarize_inputs(self):
        """
        Transforms inputs into a standard format that is used by all sub-modules.
        """
        # standardize inputs

        if is_list(self.Y):
            # check that all elements of Y are N x 1
            # assert check_shape_last_dim(self.Y, 1)
            pass
        else:
            # check that Y in  N x P
            assert len(self.Y.shape) == 2
            P = self.Y.shape[-1]
            self.Y = [self.Y[:, p][:, None] for p in range(P)]

        if is_list(self.Y) and not is_list(self.X):
            # copy X num_latents times
            P = len(self.Y)

            self.X = copy_to_list(self.X, P)

        self.Y_arr = [self.Y] if not is_list(self.Y) else self.Y
        self.kernel_arr = [self.kernel] if not is_list(self.kernel) else self.kernel
        self.likelihood_arr = (
            [self.likelihood] if not is_list(self.likelihood) else self.likelihood
        )

        self.sparsity_arr = (
            [self.sparsity] if not is_list(self.sparsity) else self.sparsity
        )

        self.num_outputs = len(self.Y_arr)

        if False:
            print("X: ", [x.shape[0] for x in self.X_arr])
            print("Y: ", [y.shape[0] for y in self.Y_arr])
            print("kernel: ", self.kernel)
            print("likelihood: ", self.likelihood)
            print("inference: ", self.inference)
            print("options: ", self.options)
            print("name: ", self.name)

    def setup(self):
        """
        Useful to allow model to have specific setups
        """
        pass

    def set_defaults_if_needed(self):
        """
        @TODO(ollie) fix default shapes or move before standarize inputs
        """
        if self.kernel is None:
            self.kernel = RBF()

        if self.likelihood is None:
            self.likelihood = GaussianLikelihood()

        if self.inference is None:
            self.inference = BatchInference(name="batch")

        if self.options is None:
            self.options = {}

        if self.name is None:
            self.name = "model"

        if "whiten" not in self.options:
            self.options["whiten"] = False

        if self.sparsity is None:
            self.sparsity = NoSparsity()

    def check_model(self):
        """
        Checks the input data, likelihoods and kernels
        """

        if self.kernel is None:
            warnings.warn("No kernel specified. Default will be used.", RuntimeWarning)

        if self.likelihood is None:
            warnings.warn(
                "No likelihood specified. Default will be used.", RuntimeWarning
            )

        if self.inference is None:
            warnings.warn(
                "No inference method specified. Default inference will be used.",
                RuntimeWarning,
            )

    def description() -> str:
        """
        Generates an automatic description of the model and inference schemed created
        """
        raise NotImplementedError()

    def anchor(self, params):
        # all_params = Parameter.ALL_PARAM_DICT.copy()
        # all_params.update(params)

        Parameter.PARAM_DICT.update(params)
        self.update(Parameter.PARAM_DICT)
        # Parameter.update(all_params)

    def print(self):
        for key, val in Parameter.PARAM_DICT.items():
            param_val = Parameter.OBJ_DICT[key]()
            if np.sum(param_val.shape) > 5:
                param_val = np.sum(param_val)

            print(key, ": transformed: ", param_val)

    @abstractmethod
    def get_objective(self):
        return

    @abstractmethod
    def predict_y(self, XS: np.ndarray, diagional_var: Optional[bool] = True):
        return

    @abstractmethod
    def predict_f(self, XS: np.ndarray, diagional_var: Optional[bool] = True):
        return

    def predict_latents(self, XS: np.ndarray, diagional_var: Optional[bool] = True):
        raise NotImplementedError(
            "predict_latents has not been implemented for this model yet or it not applicable!"
        )

    def checkpoint(self, filename="gpjax.ckpt"):
        if not filename.endswith(".ckpt"):
            filename = filename + ".ckpt"

        ckpt = {}

        params = Parameter.PARAM_DICT
        # ensure params are numpy arrays
        params = {k: onp.asarray(i) for k, i in params.items()}

        ckpt["parameters"] = params

        with open(filename, "wb") as f:
            pickle.dump(ckpt, f)

    def restore_from_checkpoint(self, filename="gpjax.ckpt"):

        if not filename.endswith(".ckpt"):
            filename = filename + ".ckpt"

        with open(filename, "rb") as f:
            ckpt = pickle.load(f)

        params = ckpt["parameters"]
        self.anchor(params)
