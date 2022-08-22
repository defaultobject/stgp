""" Spatial GP regression computed through Kalman Filtering and Smoothing"""
import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import stgp as lego
from stgp.trainers import GradDescentTrainer, ScipyTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import Matern32, SpatioTemporalSeperableKernel
from stgp.likelihood import Gaussian, ReshapedGaussian
from stgp.data import SpatioTemporalData
from stgp.transforms.sdes import LTI_SDE
from stgp.transforms import Independent
from stgp.models import GP
from stgp.sparsity import NoSparsity

import objax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import batchjax
import stdata as st
from stdata.plots import grid_to_matrix
import matplotlib.pyplot as plt
from pathlib import Path
import stdata
import pandas as pd

from data_zoo import single_output_spatial_data

# Generate data

XS, X, Y = single_output_spatial_data(20, 20, 200, 200, seed=0)

data = SpatioTemporalData(X=X, Y=Y, sort=True)

# Setup Model

lik = ReshapedGaussian(Gaussian(), num_blocks=data.Nt, block_size=data.Ns)

latents = [
    GP(
        sparsity=NoSparsity(), 
        kernel= SpatioTemporalSeperableKernel(
            Matern32(input_dim=1, lengthscales=[0.1], active_dims=[0]),
            Matern32(input_dim=1, lengthscales=[0.1], active_dims=[1])
        )
    )
]


base_gp = Independent(latents)
prior = LTI_SDE(base_gp)

m = lego.models.GP(
    data = data, 
    prior = prior,
    likelihood = lik,
    inference='Sequential'
)

print(m.get_objective())

if False:
    trainer = GradDescentTrainer(m, objax.optimizer.Adam)

    # Train
    epochs = 500
    callback = progress_bar_callback(epochs)
    learning_curve, training_time = trainer.train(
        0.01,
        epochs,
        callback = callback
    )

    # Plot learning curve
    plt.plot(learning_curve)
    plt.show()

# Plot
pred_mu, pred_var = m.predict_y(XS)

pred_mat, _ = stdata.plots.grid_to_matrix(pd.DataFrame(XS, columns=['lon', 'lat']), pred_mu)
plt.imshow(pred_mat)

plt.show()

