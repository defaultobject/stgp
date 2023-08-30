import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import stgp
from stgp.data import AggregatedData
from stgp.models import GP
from stgp.kernels import ScaleKernel, RBF
from stgp.transforms import Aggregate, Independent
from stgp.trainers.callbacks import progress_bar_callback
from stgp.trainers import GradDescentTrainer, ScipyTrainer, NatGradTrainer

from stgp.trainers.standard import VB_NG_ADAM
import objax
from tqdm import trange

def aggregate_in_time(x, y, group_size):
    """
    Groups (x, y) in groups of group_size. Y is returned as the average of the group.
    """

    N = x.shape[0]

    aggr_x = []
    aggr_y = []

    current_index = 0
    for i in range(int(N/group_size)):
        next_index = current_index + group_size
        aggr_x.append(x[current_index:next_index])

        aggr_y.append(
            np.sum(y[current_index:next_index])/(group_size)
        )

        current_index = next_index

    return np.array(aggr_x), np.array(aggr_y)

def plot_timeseries_aggregated_xy(x, y):
    linecolor= 'blue'

    for n in range(y.shape[0]):
        # get group boundary
        min_x, max_x = np.min(x[n]), np.max(x[n])
        plt.plot([min_x, max_x], [y[n], y[n]], c=linecolor)

def plot_timeseries_aggregated_pred(x, mu, var):
    linecolor= 'red'

    mu = np.squeeze(mu)
    var = np.squeeze(var)

    for n in range(x.shape[0]):
        # get group boundary
        min_x, max_x = np.min(x[n]), np.max(x[n])

        plt.plot([min_x, max_x], [mu[n], mu[n]], c=linecolor)
        plt.fill_between([min_x, max_x], [mu[n]-1.96*np.sqrt(var[n]), mu[n]-1.96*np.sqrt(var[n])], [mu[n]+1.96*np.sqrt(var[n]), mu[n]+1.96*np.sqrt(var[n])], alpha=0.4, facecolor=linecolor)

np.random.seed(0)

x = np.linspace(0, 1, 100)
f = np.sin(x*10) 

xs = np.linspace(-1, 2, 500)
fs = np.sin(xs*10) 


XS = np.linspace(-1, 2, 1000)[:, None]

x_aggr, f_aggr = aggregate_in_time(x, f, 5)
xs_aggr, _ = aggregate_in_time(xs, fs, 10)


y_aggr = f_aggr + 0.01*np.random.randn(f_aggr.shape[0])
Y_aggr = y_aggr[:, None]
X_aggr = x_aggr[..., None]

if False:
    plt.plot(x, f)
    plot_timeseries_aggregated_xy(x_aggr, y_aggr)
    plt.show()

D = 1

data = AggregatedData(X_aggr, Y_aggr)
lik = stgp.likelihood.Gaussian(0.01)
Z = stgp.sparsity.FullSparsity(Z = np.linspace(0, 1, 20)[:, None])

latent_gp = GP(
    sparsity = Z, 
    kernel = ScaleKernel(RBF(input_dim=D, lengthscales=[0.1 for d in range(D)]))
)

prior = Aggregate(Independent([latent_gp]))

m = GP(
    data = data,
    likelihood = [lik],
    prior = prior,
    inference='Variational'
)

if True:
    # Train
    epochs = 100

    trainer = VB_NG_ADAM(m)
    trainer.ng_trainer.train(1.0, 1)
    lc_arr, _ = trainer.train([0.01, 0.1], [epochs, [1, 1]], callback=progress_bar_callback(epochs))

    plt.plot(lc_arr[::2])
    plt.show()

m.print()

# Predict
pred_aggr_mu, pred_aggr_var = m.predict_f(xs_aggr[..., None], squeeze=True)
pred_mu, pred_var = m.predict_latents(XS, squeeze=True, diagonal=True)

pred_mu = np.squeeze(pred_mu)
pred_var = np.squeeze(pred_var)

# Plot results
fig = plt.figure(figsize=(10, 5))
ax = plt.gca()

ax.fill_between(np.squeeze(XS), np.squeeze(pred_mu - 2*np.sqrt(pred_var)), np.squeeze(pred_mu + 2*np.sqrt(pred_var)), alpha=0.4)
ax.plot(XS, pred_mu)

plot_timeseries_aggregated_pred(xs_aggr, pred_aggr_mu, pred_aggr_var)
plot_timeseries_aggregated_xy(x_aggr, y_aggr)
plt.show()
