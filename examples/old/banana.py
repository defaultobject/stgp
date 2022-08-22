import pandas as pd
import numpy as np
import objax

import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import matplotlib.pyplot as plt

import stgp
from stgp.models import GP
from stgp.trainers import GradDescentTrainer, ScipyTrainer, NatGradTrainer
from stgp.computation.parameter_transforms import identity, inv_probit
from stgp.transforms.basic import InvProbit
from stgp.kernels import ScaleKernel, RBF
from stgp.transforms import Independent, One2One

X_df = pd.read_csv('data/banana_X_train.csv', header=None)
Y_df = pd.read_csv('data/banana_Y_train.csv', header=None)
X_df.columns = ['x', 'y']
Y_df.columns = ['Y']

X = np.array(X_df[['x', 'y']])
Y = np.array(Y_df[['Y']])

x_grid = np.linspace(-3, 3, 40)
xx, yy = np.meshgrid(x_grid, x_grid)
XS = np.vstack((xx.flatten(), yy.flatten())).T

# Construct model

def model(X, Y):
    P = 1
    Q = 1
    D = 2

    data = stgp.data.Data(X, Y)
    lik = [stgp.likelihood.Bernoulli(link_fn=identity) for p in range(P)]

    latent_gps = [
        GP(
            sparsity=stgp.sparsity.NoSparsity(Z_ref=data._X), 
            kernel = ScaleKernel(RBF(input_dim=D, lengthscales=[1.0 for d in range(D)]))
        )
        for q in range(Q)
    ]

    prior = One2One(
        Independent(latent_gps),
        [InvProbit() for p in range(P)]
    )

    m = GP(
        data = data,
        likelihood = lik,
        prior = prior,
        inference='Variational',
        ell_samples=100,
        prediction_samples=1000
    )

    return m

m = model(X, Y)


# train
def train(m):
    m.get_objective()

    m.approximate_posterior.approx_posteriors[0]._S_chol.fix()
    m.approximate_posterior.approx_posteriors[0]._m.fix()

    ng_trainer = NatGradTrainer(m, enforce_psd_type='retraction')
    trainer = GradDescentTrainer(m, objax.optimizer.Adam)

    ng_trainer.train(0.01, 10)
    ng_trainer.train(0.1, 10)

    lc_arr = []
    num_epochs = 200
    for i in range(num_epochs):
        print(f'{i}/{num_epochs}')
        lc, _ = trainer.train(0.01, 1)
        ng_trainer.train(0.1, 1)
        lc_arr.append(lc)

    try:
        plt.plot(np.array(lc_arr).flatten())
        plt.show()
    except Exception as e:
        print(e)

train(m)

# predict
median, ci_lower, ci_upper = m.confidence_intervals(XS)

pred = median[0]

# plot
plt.scatter(XS[:, 0], XS[:, 1], c=pred)
plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0])
_ = plt.contour(
    xx,
    yy,
    pred.reshape(*xx.shape),
    [0.5],  # plot the p=0.5 contour line only
    colors="k",
    linewidths=1.8,
    zorder=100,
)
plt.show()
