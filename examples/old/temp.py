""" Single GP regression """
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
from stgp.transforms import LinearTransform, OutputMap, Identity, MultiOutput, DataLatentPermutation
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
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import stdata
import stdata as st
from stdata.plots import grid_to_matrix
from stdata.grids import create_spatial_temporal_grid, create_spatial_grid


# Fix randomness
np.random.seed(0)

Nt = 5
T = np.linspace(0, 1, Nt)

#X = create_spatial_temporal_grid(T, 0, 1, 0, 1, 5, 5)
X = create_spatial_grid(0, 1, 0, 1, 5, 35)

XS = create_spatial_grid(0, 1, 0, 1, 10, 100)

init_idx = X[:, 0] == 0
edge_idx = (X[:, 1] == 0) 

Y = np.ones([X.shape[0], 1])*np.NaN
Y[init_idx] = 0
Y[edge_idx] = 1

N = X.shape[0]

if True:
    fig, axes = plt.subplots(1, Nt, sharey=True)

    norm = mpl.colors.Normalize(0, 1)
    for i, t in enumerate(T):
        i_idx = X[:, 0] == t

        axes[i].scatter(X[i_idx, 1], Y[i_idx, 0], norm=norm)
        axes[i].set_ylim(-0.5, 1.5)
    plt.show()
    exit()

Y = np.hstack([Y, np.zeros_like(Y)])

data = stgp.data.Data(X, Y)

base_kernel_2d = RBF(input_dim = 2, lengthscales = [0.1, 0.1])

base_kernel_2d.lengthscale_param.fix()

diff_op_prior = DifferentialOperatorJoint(
    GP(
        sparsity=stgp.sparsity.NoSparsity(Z=X), 
        kernel = base_kernel_2d
    ),
    SecondOrderDerivativeKernel_2D(base_kernel_2d)
)

# required when using a full approximate posterior because the prior is defined in latent-data format
#   however when computing expected log likelihoods and predictions everything is in data-latent format

#print(diff_op_prior.mean_blocks(X).shape)
#print(diff_op_prior.b_mean_blocks(X[None, ...]).shape)
#print(diff_op_prior.mean(X).shape)
#print(diff_op_prior.b_mean(X[None, ...]).shape)
#print(diff_op_prior.np_mean(X).shape)

prior_output_1, prior_output_2 = OutputMap(
    diff_op_prior, 
    [[0], [0, 1, 2, 3, 4]], 
)

pde_output = HeatEquation2D(prior_output_2)


if False:
    print(pde_output.mean(X).shape)
    print(pde_output.covar(X, XS).shape)

    print(prior_output_1.mean(X).shape)
    print(prior_output_1.covar(X, XS).shape)

prior = MultiOutput([
    prior_output_1,
    pde_output
])

# Defined in data-latent format?
q = FullGaussianApproximatePosterior(dim = N * diff_op_prior.output_dim)

lik_arr = [Gaussian(variance=0.01), Gaussian(variance=0.001)]
lik_arr[0].variance_param.fix()
lik_arr[1].variance_param.fix()

# Create Model
m = stgp.models.GP(
    data = data,
    prior = prior,
    likelihood = lik_arr,
    inference='Variational',
    approximate_posterior = q,
    prediction_samples = 1000
)

pred_mu, pred_var = m.predict_f(X)

#print(pred_mu.shape)

if False:
    # Train
    epochs = 1

    callback = progress_bar_callback(epochs)

    learning_curve, training_time = GradDescentTrainer(
        m, 
        objax.optimizer.Adam,
    ).train(
        0.01,
        epochs,
        callback = callback
    )

    # Plot learning curve
    if False:
        print(learning_curve[0], learning_curve[-1])
        plt.plot(learning_curve)
        plt.show()

print(m.get_objective())

if True:
    ng_trainer = NatGradTrainer(m)
    ng_trainer.train(0.1, 100) 
    ng_trainer.train(0.5, 10) 
    ng_trainer.train(1.0, 1) 

print(m.get_objective())

pred_mu, pred_var = m.predict_f(XS)

if False:
    print(learning_curve[0], learning_curve[-1])
    plt.plot(learning_curve[100:])
    plt.show()

pred_mu = pred_mu[0]

if True:
    fig, axes = plt.subplots(1, Nt, sharey=True)

    norm = mpl.colors.Normalize(0, 1)
    for i, t in enumerate(T):
        i_idx = XS[:, 0] == t

        axes[i].plot(XS[i_idx, 1], pred_mu[i_idx])
        axes[i].set_ylim(-0.5, 1.5)
    plt.show()


