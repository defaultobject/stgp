""" Batch Gaussian Process Regression """

import sys
sys.path.append('../../')

import jax
jax.config.update("jax_enable_x64", True)
import objax
import numpy as np

from example_utils.data_zoo import single_output_timeseries
from example_utils import colors
from stgp.trainers import ScipyTrainer, GradDescentTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels.matern import ScaledMatern32, Matern32
from stgp.kernels.arccosine import ArcCosine
from stgp.likelihood import Gaussian

import stgp
from stgp.models import GP

import matplotlib.pyplot as plt

import jaxopt
from jax.flatten_util import ravel_pytree


from stgp.trainers.jaxopt import JaxoptTrainer

stgp.settings.linear_solver = stgp.settings.SolveType.CG

# Construct Data
XS, X, Y = single_output_timeseries(100, 100, seed=0)

K = ArcCosine(order=0)
#K = Matern32(input_dim=1, lengthscales=[0.1])

Y[40:50] = np.NaN

if True:
    print(K.K(X, X))
    plt.imshow(K.K(X, X))
    plt.show()


# Construct Model
m = GP(X, Y, kernel=[K], likelihood=Gaussian(variance=0.1))

m.print()

pred_mu, pred_var = m.predict_y(XS)

pred_mu = np.squeeze(pred_mu)
pred_var = np.squeeze(pred_var)

plt.fill_between(
    np.squeeze(XS), 
    np.squeeze(pred_mu - 1.96*np.sqrt(pred_var)), 
    np.squeeze(pred_mu + 1.96*np.sqrt(pred_var)), 
    facecolor=colors.LINE_COL, 
    alpha=0.3
)


plt.plot(XS, pred_mu, color=colors.LINE_COL, label='GP Fit')
plt.scatter(X, Y, color='black', label='Training Data')
plt.legend()
plt.show()

