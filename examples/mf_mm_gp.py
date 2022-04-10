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
from legogp.data import Data, TemporalData, MultiOutputTemporalData, get_sequential_data_obj, SpatioTemporalData, DataReshape
from legogp.sparsity import NoSparsity, StackedNoSparsity, SpatialSparsity
from legogp.approximate_posteriors import MeanFieldApproximatePosterior , MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian
from legogp.models import GP
from legogp.transforms import DataLatentPermutation , Independent

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


X1 = X
X2 = X[20:]
Y1 = Y[:, 0][:, None]
Y2 = Y[20:, 1][:, None]

m1 = GP(
    data = Data(X1, Y1),
    kernel = RBF(),
    inference='Batch'
)

m2 = GP(
    data = Data(X2, Y2),
    kernel = DeepRBF(parent=m1),
    inference='Batch'
)
print(m1.get_objective())
print(m2.get_objective())

m1.predict_y(X2)
m2.predict_y(X2)
breakpoint()
