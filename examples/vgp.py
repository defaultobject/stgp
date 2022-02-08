import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import legogp as lego
from legogp.trainers import SimpleTrainer, ScipyTrainer, NatGradTrainer, SwitchTrainer, GradDescentTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.utils.utils import match_suffix

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
import jax.tools.jax_to_hlo
from jax.lib import xla_client
import jax.profiler


checkpoint_folder = Path('checkpoints')
checkpoint_folder.mkdir(exist_ok=True)

#jax.profiler.start_trace("/tmp/tensorboard")


# generate data
P = 1

N = 1000000
M = 500

XS = np.linspace(-2, 3, 500)[:, None]
x = np.linspace(0, 1, N)
y1 = np.sin(x*10)+0.01*np.random.randn(N)

X = x[:, None]
Y1 = y1[:, None]

Y = np.hstack([Y1 for p in range(P)])+2.0

ys = XS

assert Y.shape[1] == P

qu = lego.approximate_posteriors.MeanFieldApproximatePosterior(dim_list=[M])

Z = np.linspace(0, 1, M)[:, None]

m1 = lego.models.GP(
    X,
    Y,
    Z = Z,
    inference='Variational',
    whiten=False,
    minibatch_size=500,
    kernel = lego.kernels.ScaleKernel(lego.kernels.RBF(lengthscales=[0.5])) + lego.kernels.ScaleKernel(lego.kernels.BiasKernel()),
    likelihood = lego.likelihood.Gaussian(0.1),
    approximate_posterior = qu
)

m_name = match_suffix('._m', m1.vars().keys())
s_chol_name = match_suffix('._S_chol', m1.vars().keys())
approx_posterior_vars = [m_name, s_chol_name]


grad_step = GradDescentTrainer(m1, objax.optimizer.Adam, hold_vars = approx_posterior_vars)
nat_grad_step = NatGradTrainer(m1, schedule='log')
nat_grad_constant_step = NatGradTrainer(m1, schedule='constant')

beta_init = 1e-4
beta_final=0.01
K = 5
trainer = SwitchTrainer(
    [nat_grad_step, grad_step],
    1,
    [[beta_init, beta_final], 1e-3],
    [K, 1],
    None
)
learning_curve_1, training_time = trainer.train()

trainer = SwitchTrainer(
    [nat_grad_constant_step, grad_step],
    100,
    [beta_final, 1e-3],
    [1, 1],
    None
)
learning_curve_2, training_time = trainer.train()

breakpoint()


m = lego.models.GP(
    XS, ys, 
    kernel=lego.kernels.deep_kernels.DeepRBF(parent=m1)
)


restore = False

if restore:
    m.load_from_checkpoint(str(checkpoint_folder / 'vgp'))
else:
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

    print(learning_curve[0], learning_curve[-1])

    m.checkpoint(str(checkpoint_folder / 'vgp'))

pred_mu, pred_var = m.predict_f(XS, squeeze=True)

fig = plt.figure()
plt.fill_between(
    np.squeeze(XS),
    np.squeeze(pred_mu) + 2*np.squeeze(np.sqrt(pred_var)),
    np.squeeze(pred_mu) - 2*np.squeeze(np.sqrt(pred_var)),
    alpha = 0.4
)
plt.plot(XS, pred_mu)
plt.scatter(X, Y, c='black')
plt.show()
