# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
#       jupytext_version: "1.16.0"
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Batch Gaussian Process Regression (STGP + JAX)
#
# This example fits a 1D Gaussian Process to a synthetic single-output time series and plots the posterior mean with a 95% confidence band.

# %% [markdown]
# ## Imports and configuration

# %%
import jax
import numpy as np
import matplotlib.pyplot as plt

from stgp.models import GP
from stgp.kernels.matern import Matern32
from stgp.trainers import ScipyTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.utils.example_utils.data_zoo import single_output_timeseries
from stgp.utils.example_utils import colors

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Construct data
#
# - `X, Y` are training inputs/targets
# - `XS` is a grid of test inputs for plotting predictions

# %%
XS, X, Y = single_output_timeseries(100, 100, seed=0)

# %% [markdown]
# ## Define kernel and model
#
# We use a Matérn 3/2 kernel. The model uses the default Gaussian likelihood.

# %%
K = Matern32(input_dim=1, lengthscales=[0.1], mask=[1])

m = GP(X, Y, kernel=[K])
m.print()

# %% [markdown]
# ## Objective before training

# %%
print(m.get_objective())

# %% [markdown]
# ## Train (L-BFGS-B via SciPy)

# %%
max_iters = 100
trainer = ScipyTrainer(m, "L-BFGS-B")
trainer.train(None, max_iters, callback=progress_bar_callback(max_iters))

# %% [markdown]
# ## Objective and metrics after training

# %%
print(m.get_objective())
print("NLPD:", m.nlpd(X, Y))

# %% [markdown]
# ## Predict on a grid
#
# We'll request the full predictive covariance from `predict_y`, then plot the mean and marginal variance.

# %%
pred_mu, pred_var = m.predict_y(XS)

pred_mu = np.squeeze(pred_mu)
pred_var = np.squeeze(pred_var)

print("pred_mu shape:", pred_mu.shape)
print("pred_var shape:", pred_var.shape)

# %% [markdown]
# ## Plot posterior mean and 95% interval

# %%
XS_plot = np.squeeze(XS)

plt.fill_between(
    XS_plot,
    pred_mu - 1.96 * np.sqrt(pred_var),
    pred_mu + 1.96 * np.sqrt(pred_var),
    facecolor=colors.LINE_COL,
    alpha=0.3,
    label="95% CI",
)

plt.plot(XS_plot, pred_mu, color=colors.LINE_COL, label="GP fit")
plt.scatter(np.squeeze(X), np.squeeze(Y), color="black", label="Training data")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()
