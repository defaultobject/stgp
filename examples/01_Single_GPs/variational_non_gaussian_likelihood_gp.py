""" Variational Gaussian Process Regression with a Bernoulli Likelihood"""
import sys
sys.path.append('../')

import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', True)
import objax
import numpy as np

from example_utils.data_zoo import single_output_timeseries
from example_utils import colors
from stgp.trainers import ScipyTrainer, GradDescentTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
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

print(f'X: {X.shape}, Y: {Y.shape}')

# Construct Model
data = Data(X, Y)
m = GP(
    data = data,
    prior = One2One(
        Independent([
            GP(
                sparsity = stgp.sparsity.NoSparsity(Z = data.X), 
                kernel = RBF(input_dim=30),
                prior = True
            )
        ]),
        [InvProbit()],
    ),
    likelihood = ProductLikelihood([Bernoulli(link_fn=identity)]),
    inference='Variational',
    ell_samples=10,
    prediction_samples=1
)

# Train
max_iters = 10

ng_trainer = NatGradTrainer(m, enforce_psd_type='laplace_gauss_newton')
m.approximate_posterior.fix()

trainer = GradDescentTrainer(m, objax.optimizer.Adam)

#ng_trainer.train(0.01, 10)
for i in trange(max_iters):
    trainer.train(0.01, 1)
    #ng_trainer.train(0.1, 1)

# Predict
pred_mu, pred_var = m.predict_f(X)

# Compute a ROC curve
fpr, tpr, _ = roc_curve(np.squeeze(Y), np.squeeze(pred_mu))

plt.plot(fpr, tpr)
plt.show()
