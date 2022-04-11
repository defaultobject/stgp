"""
This files show how to construct low level multi-task variational models in the
    1) temporal and spatio-temporal setting (--time, --st)
    2) mean-field and dense posterior setting (--mf, --fp)
    3) batch / sde CVI model with no sparsity (--batch, --sde) (--no-Z)
    4) sde CVI model with spatial sparsity (--spatial-Z)
"""
import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import legogp as lego
from legogp.trainers import SimpleTrainer, NatGradTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.kernels import Matern32, SpatioTemporalSeperableKernel, RBF
from legogp.kernels.deep_kernels import DeepRBF
from legogp.likelihood import Gaussian, BlockDiagonalGaussian, ReshapedBlockDiagonalGaussian
from legogp.data import Data, TemporalData, MultiOutputTemporalData, get_sequential_data_obj, SpatioTemporalData, DataReshape, TransformedData
from legogp.sparsity import NoSparsity, StackedNoSparsity, SpatialSparsity
from legogp.approximate_posteriors import MeanFieldApproximatePosterior , MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian
from legogp.models import GP
from legogp.transforms import DataLatentPermutation , Independent
from legogp.transforms.multi_output import LMC
from legogp.core import MultiObjectiveModel
from legogp.metrics.nlpd import nlpd
from legogp.transforms.basic import Log

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

from data_zoo import multi_output_timeseries

P = 6
XS, X, Y = multi_output_timeseries(P, 100, 1000, seed=0)

Y = np.square(Y)

Y1 = Y[:, :3]
Y2 = Y[:, :3]

P1 = 3
P2 = 3

m1_latents = [
    GP(
        data = Data(X, Y[:, p][:, None]),
        kernel = RBF(lengthscales=[0.1]),
        likelihood = [Gaussian(variance=0.1)],
        inference='Batch'
    )
    for p in range(P1)
]

m2_prior = LMC(m1_latents, output_dim = P2)
data = TransformedData(Data(X, Y2), [Log(), Log(), Log()])

m2 = GP(
    data = data, 
    prior = m2_prior,
    likelihood = [Gaussian(variance=0.1) for p in range(P2)],
    inference='Batch'
)


#print(m2.get_objective())
#print(nlpd(X, Y2, m2))

m = MultiObjectiveModel([m2] + m1_latents)

if True:
    # Train
    epochs = 500
    callback = progress_bar_callback(epochs)
    learning_curve, training_time = SimpleTrainer().train(
        m, 
        objax.optimizer.Adam,
        0.01,
        epochs,
        callback = callback
    )

    # Plot learning curve
    plt.plot(learning_curve)
    plt.show()

print(nlpd(X, Y2, m2))

median, lower_ci, upper_ci = m2.confidence_intervals(XS)

#pred_mu_1, pred_var_1 = m1.predict_y(XS)
#pred_mu_2, pred_var_2 = m2.predict_y(XS)

#plt.scatter(X1, Y1)
#plt.plot(XS, pred_mu_1)

for p in range(P2):
    plt.scatter(X, Y[:, p])
    plt.plot(XS, median[p])

plt.show()
