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
from legogp.kernels import Matern32, SpatioTemporalSeperableKernel, RBF, ScaleKernel
from legogp.kernels.deep_kernels import DeepRBF, DeepLinear
from legogp.likelihood import Gaussian, BlockDiagonalGaussian, ReshapedBlockDiagonalGaussian
from legogp.data import Data, TemporalData, MultiOutputTemporalData, get_sequential_data_obj, SpatioTemporalData, DataReshape, TransformedData
from legogp.sparsity import NoSparsity, StackedNoSparsity, SpatialSparsity
from legogp.approximate_posteriors import MeanFieldApproximatePosterior , MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian
from legogp.models import GP
from legogp.transforms import DataLatentPermutation , Independent
from legogp.core import MultiObjectiveModel
from legogp.metrics.nlpd import nlpd
from legogp.transforms.basic import Log, Softminus, Affine, ReverseFlow

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


P = 2
XS, X, Y = multi_output_timeseries(P, 100, 1000, seed=0)
Y = np.exp(Y)


X1 = X
X2 = X[20:]
Y1 = Y[:, 0][:, None]
Y2 = Y[20:, 1][:, None]

#plt.scatter(X1, Y1)
#plt.scatter(X2, Y2)
#plt.show()

#data = Data(X1, Y1)
data = TransformedData(Data(X1, Y1), [ReverseFlow(Affine(np.std(Y1), np.mean(Y1), train=False))])
m1 = GP(
    data = data,
    kernel = ScaleKernel(RBF(lengthscales=[0.01])),
    likelihood = [Gaussian(variance=0.1)],
    inference='Batch'
)

data = TransformedData(Data(X2, Y2), [Softminus()])
#data = Data(X2, Y2)

m2 = GP(
    data = data,
    kernel = ScaleKernel(DeepRBF(parent=m1, lengthscale=[1.0])) + ScaleKernel(DeepLinear(parent=m1)),
    likelihood = [Gaussian(variance=0.1)],
    inference='Batch'
)

m = MultiObjectiveModel([m2, m1])

YS = Y[:, 1][:, None]

print(nlpd(X, YS, m2))

if True:
    # Train
    epochs = 1000
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

print(nlpd(X, YS, m2))

median_1, lower_ci_1, upper_ci_1 = m1.confidence_intervals(XS)
median_2, lower_ci_2, upper_ci_2 = m2.confidence_intervals(XS)

plt.scatter(X1, Y1)
plt.fill_between(np.squeeze(XS), lower_ci_1[0], upper_ci_1[0], alpha=0.4)
plt.plot(XS, median_1[0])

plt.scatter(X2, Y2)
plt.fill_between(np.squeeze(XS), lower_ci_2[0], upper_ci_2[0], alpha=0.4)
plt.plot(XS, median_2[0])
plt.show()
