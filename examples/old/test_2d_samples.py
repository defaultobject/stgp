import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import stgp
from stgp.trainers import GradDescentTrainer, ScipyTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel
from stgp.kernels.diff_op import SecondOrderDerivativeKernel_2D, SecondOrderDerivativeKernel_1D
from stgp.likelihood import Gaussian
from stgp.models import GP
from stgp.transforms import LinearTransform, OutputMap, MultiOutput, DataLatentPermutation
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
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import stdata
import stdata as st
from stdata.plots import grid_to_matrix
from stdata.grids import create_spatial_temporal_grid, create_spatial_grid


Ns, Nt = 20, 20
X = create_spatial_grid(0, 1, 0, 1, Ns, Nt)

base_kernel_2d = RBF(input_dim = 2, lengthscales = [0.1, 0.1])

diff_op_prior = DifferentialOperatorJoint(
    GP(
        sparsity=stgp.sparsity.NoSparsity(Z=X), 
        kernel = base_kernel_2d
    ),
    SecondOrderDerivativeKernel_2D(base_kernel_2d)
)

Kxx = diff_op_prior.covar(X, X)

if False:
    plt.imshow(Kxx)
    plt.show()

N = X.shape[0]

# check out samples
latent = np.random.multivariate_normal(np.zeros(Kxx.shape[0]), Kxx)

# there are three outputs
latent_t = latent[:N].reshape([Ns, Nt])
latent_dt = latent[N:N*2].reshape([Ns, Nt])
latent_dt2 = latent[N*2:N*3].reshape([Ns, Nt])
latent_dx = latent[N*3:N*4].reshape([Ns, Nt])
latent_dx2 = latent[N*4:N*5].reshape([Ns, Nt])

test_dt_gradient, test_dx_gradient = np.gradient(latent_t, 1/Nt, 1/Ns)
test_dt2_gradient, _ = np.gradient(latent_dt, 1/Nt, 1/Ns)
_, test_dx2_gradient = np.gradient(latent_dx, 1/Nt, 1/Ns)


fig, axes = plt.subplots(5, 3)

axes[0][0].set_title('Kernel')
axes[0][0].imshow(latent_t)
axes[1][0].imshow(latent_dt)
axes[2][0].imshow(latent_dt2)
axes[3][0].imshow(latent_dx)
axes[4][0].imshow(latent_dx2)

axes[0][1].set_title('np approximation')
axes[0][1].imshow(latent_t)
axes[1][1].imshow(test_dt_gradient)
axes[2][1].imshow(test_dt2_gradient)
axes[3][1].imshow(test_dx_gradient)
axes[4][1].imshow(test_dx2_gradient)

axes[0][2].set_title('abs error')
axes[0][2].imshow(np.abs(latent_t - latent_t ))
axes[1][2].imshow(np.abs(latent_dt - test_dt_gradient ))
axes[2][2].imshow(np.abs(latent_dt2 - test_dt2_gradient ))
axes[3][2].imshow(np.abs(latent_dx - test_dx_gradient ))
axes[4][2].imshow(np.abs(latent_dx2 - test_dx2_gradient ))
plt.show()
