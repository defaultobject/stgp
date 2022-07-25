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
import matplotlib.pyplot as plt
from pathlib import Path

# Fix randomness
np.random.seed(0)

XS, X, Y = single_output_spatial_data(10, 10, 30, 30, seed=0)
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

A = pde_output.covar(X, X)
B = diff_op_prior.covar(X, X)

np.linalg.cholesky(B+1e-5 * np.eye(B.shape[0]))


if False:
    print(pde_output.mean(X))
    print(pde_output.covar(X, X))

    plt.imshow(pde_output.covar(X, X))
    plt.show()
    breakpoint()

pde_output = DataLatentPermutation(pde_output)

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

# Create Model
m = stgp.models.GP(
    data = data,
    prior = pde_output,
    likelihood = [Gaussian(variance=0.1)],
    inference='Variational',
    approximate_posterior = q,
    prediction_samples = 100
)


print(m.predict_f(X))
print(m.predict_f(XS))
print(m.get_objective())
exit()


if True:
    # Train
    epochs = 200

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
    print(learning_curve[0], learning_curve[-1])
    plt.plot(learning_curve)
    plt.show()

print(m.predict_f(XS))

breakpoint()
