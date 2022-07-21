""" Single GP regression """
import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import stgp
from stgp.trainers import GradDescentTrainer, ScipyTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel
from stgp.kernels.diff_op import SecondOrderDerivativeKernel_2D
from stgp.likelihood import Gaussian
from stgp.models import GP
from stgp.transforms import LinearTransform, OutputMap, Identity, MultiOutput
from stgp.transforms.basic import InputMeanFunction
from stgp.core.model_types import get_model_type
from stgp.transforms.pdes import DifferentialOperatorJoint, HeatEquation2D
from stgp.approximate_posteriors import MeanFieldApproximatePosterior, MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian, FullGaussianApproximatePosterior

from data_zoo import single_output_spatial_data

import objax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import batchjax
import matplotlib.pyplot as plt
from pathlib import Path


# Fix randomness
np.random.seed(0)

XS, X, Y = single_output_spatial_data(20, 20, 30, 30, seed=0)
N = X.shape[0]

data = stgp.data.Data(X, Y)

base_kernel_2d = RBF(input_dim = 2, lengthscales = [1.0, 1.0])

diff_op_prior = DifferentialOperatorJoint(
    GP(
        sparsity=stgp.sparsity.NoSparsity(Z=X), 
        kernel = base_kernel_2d
    ),
    SecondOrderDerivativeKernel_2D(base_kernel_2d)
)

prior_output_1, prior_output_2 = OutputMap(
    diff_op_prior, 
    [[0], [0, 1, 2, 3, 4]], 
)

pde_output = HeatEquation2D(prior_output_2)

print(pde_output.mean(X).shape)
print(pde_output.covar(X, XS).shape)
breakpoint()

print(prior_output_1.mean(X).shape)
print(prior_output_1.covar(X, XS).shape)

prior = MultiOutput([
    prior_output_1,
    pde_output
])

q = FullGaussianApproximatePosterior(dim = N * diff_op_prior.output_dim)

# Create Model
m = stgp.models.GP(
    data = data,
    prior = prior_output_1,
    likelihood = [Gaussian(variance=0.1)]
)

print(m.get_objective())
breakpoint()
