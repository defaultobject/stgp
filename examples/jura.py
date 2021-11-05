import jax
import objax
import numpy as np
import pandas as pd

import gpax
from gpax.trainers import SimpleTrainer
from gpax.trainers.callbacks import progress_bar_callback
from gpax.kernels.deep_kernels import DeepRBF
from gpax.kernels import RBF
from gpax.models import GP

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.set_printoptions(linewidth=200)

np.random.seed(0)

# load jura dataset
jura_pred = pd.read_csv('/Users/ohamelijnck/Documents/projects/turing-HY/code/datasets/jura/downloaded_data/jura_pred.csv')

tasks = ['Cd', 'Co', 'Cr', 'Cu', 'Ni', 'Pb', 'Zn']

if False:
    fig, axes = plt.subplots(1, len(tasks))
    for i, t in enumerate(tasks):
        axes[i].set_title(t)
        axes[i].scatter(jura_pred['lat'], jura_pred['long'], c=jura_pred[t])

    plt.show()

X = np.array(jura_pred[['lat', 'long']])
Y = np.array(jura_pred[tasks])

print(f'X: {X.shape}, Y: {Y.shape}')

# Create prediction grid
lat_min, lat_max = jura_pred['lat'].min(), jura_pred['lat'].max()
lon_min, lon_max = jura_pred['long'].min(), jura_pred['long'].max()

lat_xs = np.linspace(lat_min, lat_max, 20)
lon_xs = np.linspace(lon_min, lon_max, 20)
XS = np.array([[lat, lon] for lat in lat_xs for lon in lon_xs])

# Set up model

P = len(tasks)
Q = P
latents = [
    GP(X=X, kernel=RBF(input_dim=2, lengthscales=[0.1, 0.1]), latent=True)
    for q in range(Q)
]
prior = gpax.transforms.multi_output.LMC_Unit_Tri(latents, output_dim = P)
m = GP(X=X, Y = Y, prior=prior, inference='Batch')

# Train model

if False:
    epochs = 500

    callback = progress_bar_callback(epochs)

    learning_curve, training_time = SimpleTrainer().train(
        m, 
        objax.optimizer.Adam,
        0.01,
        epochs,
        callback = callback
    )

    plt.plot(learning_curve)
    plt.show()

    m.checkpoint('lmc_model')
else:
    m.load_from_checkpoint('lmc_model')

pred_mu, pred_var = m.predict(XS)

if True:
    fig, axes = plt.subplots(1, len(tasks))
    for i, t in enumerate(tasks):
        norm = matplotlib.colors.Normalize(np.min(Y[:, i]), np.max(Y[:, i]))
        axes[i].set_title(t)
        axes[i].scatter(XS[:, 0], XS[:, 1], c=pred_mu[i], norm=norm)

        axes[i].scatter(X[:, 0], X[:, 1], c=Y[:, i], edgecolor='black', norm=norm)

    plt.show()


