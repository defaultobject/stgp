""" Variational Gaussian Process Regression with a Bernoulli Likelihood"""
import sys
sys.path.append('../')

import jax
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import objax
import numpy as np

from example_utils.data_zoo import single_output_timeseries
from example_utils import colors
from stgp.trainers import ScipyTrainer, GradDescentTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.trainers.standard import ADAM, VB_NG_ADAM
from stgp.kernels import RBF 
from stgp.likelihood import Bernoulli, ProductLikelihood
from stgp.data import Data
from stgp.transforms import Independent
from tqdm import trange

from stgp.transforms import One2One
from stgp.transforms.basic import InvProbit
from stgp.computation.parameter_transforms import identity
import stgp
from stgp.models import GP

import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_curve

X_df, Y_df = load_breast_cancer(return_X_y=True, as_frame=True)
X = np.array(X_df)
Y = np.array(Y_df)[:, None]

x = np.linspace(0, 10)
y = np.heaviside(np.sin(x), 0).astype(np.float32)
X = x[:, None]
Y = y[:, None]

print(f'X: {X.shape}, Y: {Y.shape}')

# Construct Model
data = Data(X, Y)
m = GP(
    data = data,
    prior = One2One(
        Independent([
            GP(
                sparsity = stgp.sparsity.NoSparsity(Z = data.X), 
                kernel = RBF(input_dim=X.shape[1]),
                prior = True
            )
        ]),
        [InvProbit()],
    ),
    likelihood = ProductLikelihood([Bernoulli(link_fn=identity)]),
    inference='Variational',
    ell_samples=10,
    prediction_samples=1,
    whiten=True
)

# Train
max_iters = 200

if True:
    trainer = VB_NG_ADAM(m, enforce_psd_type='laplace_gauss_newton')

    lc_arr,  _ = trainer.train([0.01, 0.1], [max_iters, [1, 1]], callback= progress_bar_callback(max_iters))

    if False:
        plt.plot(lc_arr[::2])
        plt.show()

else:
    trainer = ADAM(m)
    lc_arr, _ = trainer.train(0.01, max_iters, callback=progress_bar_callback(max_iters))
    plt.plot(lc_arr)
    plt.show()

samples = m.samples(X, num_samples=10)
samples = np.squeeze(samples)

for s in range(10):
    plt.plot(samples[s])

plt.show()

post = m.confidence_intervals(X)

print(np.squeeze(Y)-np.squeeze(post[1]))

# Predict
pred_mu, pred_var = m.predict_f(X)

# Compute a ROC curve
fpr, tpr, _ = roc_curve(np.squeeze(Y), np.squeeze(pred_mu))

plt.plot(fpr, tpr)
plt.show()
