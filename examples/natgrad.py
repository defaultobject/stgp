import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import legogp
import legogp as lego
from legogp.trainers import SimpleTrainer, ScipyTrainer, NatGradTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.approximate_posteriors import MeanFieldApproximatePosterior, MM_GaussianInnerLayerApproximatePosterior , MeanFieldConjugateGaussian, FullConjugateGaussian
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

def train_adam(m_arr, epochs):
    callback = progress_bar_callback(epochs)
    learning_curve, training_time = SimpleTrainer().train(
        m_arr, 
        objax.optimizer.Adam,
        0.01,
        epochs,
        callback = callback
    )

    print(learning_curve[0], learning_curve[-1])

    if True:
        plt.plot(learning_curve)
        plt.show()


checkpoint_folder = Path('checkpoints')
checkpoint_folder.mkdir(exist_ok=True)

# generate data
P = 2
Q = P

N = 100

XS = np.linspace(-1, 2, 1000)[:, None]
x = np.linspace(0, 1, N)
y1 = np.sin(x*10)+0.01*np.random.randn(N)

X = x[:, None]
Y1 = y1[:, None]

Y = np.hstack([(p+1)*Y1 for p in range(P)])

assert Y.shape[1] == P

legogp.settings.jitter = 1e-5

K = [lego.kernels.ScaleKernel(lego.kernels.RBF(lengthscales=[0.1])) for q in range(Q)]

if True:
    if False:
        q = MeanFieldConjugateGaussian([
            legogp.approximate_posteriors.ConjugateGaussian(
                X=X,
                surrogate_model = lambda X, Y, likelihood:  lego.models.GP(X, Y, kernel=K[q], likelihood=likelihood) # batch gp surrogate model
            )
            for q in range(Q)
        ])
    else:
        q = FullConjugateGaussian(
            X = X,
            num_latents=Q,
            surrogate_model = lambda X, Y, likelihood:  lego.models.GP(X, Y, kernel=K, likelihood=likelihood) # batch gp surrogate model
        )
else: 
    if True:
        q = lego.approximate_posteriors.FullGaussianApproximatePosterior(dim=2 * X.shape[0])
    else:
        q = None # Mean field

m = lego.models.GP(
    X,
    Y,
    inference='Variational',
    whiten=False,
    minibatch_size=None,
    kernel = K,
    likelihood = [lego.likelihood.Gaussian(0.1) for p in range(P)],
    approximate_posterior = q,
    ell_samples=100,
    prediction_samples=1000
)

print(m.get_objective())
print(m.natural_gradients(0.1))


if True:
    natgrad_trainer = NatGradTrainer(m, schedule=None)
    natgrad_trainer.train(0.1, 20)
else:
    train_adam(m, 500)


print('OBJ: ', m.get_objective())
breakpoint()

pred_mu, pred_var = m.predict_f(XS, squeeze=True)

print(np.sum(pred_mu), np.sum(pred_var))

fig = plt.figure()

for p in range(P):
    if True:
        plt.fill_between(
            np.squeeze(XS),
            np.squeeze(pred_mu[p]) + 2*np.squeeze(np.sqrt(pred_var[p])),
            np.squeeze(pred_mu[p]) - 2*np.squeeze(np.sqrt(pred_var[p])),
            alpha = 0.4
        )
    plt.plot(XS, pred_mu[p])
    plt.scatter(X, Y[:, p], c='black')
plt.show()
