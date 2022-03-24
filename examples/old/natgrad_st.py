import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', True)

import sys

import legogp
import legogp as lego
from legogp.trainers import SimpleTrainer, ScipyTrainer, NatGradTrainer
from legogp.trainers.callbacks import progress_bar_callback
from legogp.approximate_posteriors import MeanFieldApproximatePosterior, MM_GaussianInnerLayerApproximatePosterior , MeanFieldConjugateGaussian, FullConjugateGaussian
from legogp.transforms import Independent, DataLatentPermutation
from legogp.computation.permutations import data_order_to_output_order
from legogp.computation.matrix_ops import add_jitter
import objax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import batchjax
import matplotlib.pyplot as plt
from pathlib import Path

import argparse

parser = argparse.ArgumentParser()
parser.add_argument( '--cvi', action='store_true')
parser.add_argument( '--vi', action='store_true')
parser.add_argument( '--fp', action='store_true')
parser.add_argument( '--mf', action='store_true')
parser.add_argument( '--no-Z', action='store_true')
parser.add_argument( '--dense-Z', action='store_true')
parser.add_argument( '--spatial-Z', action='store_true')

cmd_args = vars(parser.parse_args())

print(cmd_args)

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
np.random.seed(0)

# generate data

def create_grid(x1, x2, y1, y2, n1=10, n2=10):
    y = np.linspace(y1, y2, n2)
    x = np.linspace(x1, x2, n1)

    grid = []
    for i in x:
        for j in y:
            grid.append([i, j])

    return np.array(grid)

def cross_product(x_t, x_s):
    return np.vstack([np.hstack([np.tile(x_t[i], [x_s.shape[0], 1]), x_s]) for i in range(x_t.shape[0])])


Nt_train = 10
Ns = 8

X = create_grid(-1, 1, -1, 1, Nt_train, Ns)
N = X.shape[0]

P = 2
Q = P

y1 = np.square(np.sin(X[:, 0]*10)+0.01*np.random.randn(N))+1
y2 = -np.square(np.cos(X[:, 1]*10)+0.01*np.random.randn(N)) - 1

Y1 = y1[:, None]
Y2 = y2[:, None]

Y = np.hstack([Y1, Y2])

assert Y.shape[1] == P

X_time = np.unique(X[:, 0])[:, None]
X_space = X[:Ns, 1:]
Y = np.reshape(Y, [Nt_train, Ns, -1])

# Convert data to st-data
data = lego.data.SpatioTemporalData(
    X_time = X_time,
    X_space = X_space,
    Y = Y
)
legogp.settings.jitter = 1e-5

K = [lego.kernels.ScaleKernel(lego.kernels.RBF(input_dim=2, lengthscales=[0.1, 0.1])) for q in range(Q)]


if cmd_args['no_Z']:
    print('--------- No Sparsity --------')

    M = N
    Z = X

    prior = lego.transforms.Independent([
        lego.models.GP(
            X=X, 
            sparsity=lego.sparsity.NoSparsity(Z=X),
            kernel=K[q]
        ) for q in range(Q)
    ])
elif cmd_args['dense_Z']:
    M = 20
    Z = X[:M, :]

    print('--------- Dense Sparsity --------')
    prior = lego.transforms.Independent([
        lego.models.GP(
            X=Z, 
            sparsity=lego.sparsity.FullSparsity(Z=Z),
            kernel=K[q]
        ) for q in range(Q)
    ])


elif cmd_args['spatial_Z']:
    M = 3
    Z_space = X[:M, 1][:, None]
    Z_time = np.unique(X[:, 0])[:, None]
    Z = cross_product(Z_time, Z_space)
    Z_all = None


    print('--------- Spatial Sparsity --------')
    prior = lego.transforms.Independent([
        lego.models.GP(
            X=Z, 
            sparsity=lego.sparsity.SpatialSparsity(Z=Z_space),
            kernel=K[q]
        ) for q in range(Q)
    ])

if cmd_args['fp']:
    prior = DataLatentPermutation(prior)

Z_all = np.tile(Z, [Q, 1, 1])

if cmd_args['mf']:
    print('--------- MF --------')

    if cmd_args['no_Z']:
        block_size = 1

    elif cmd_args['dense_Z']:
        block_size = M 


    elif cmd_args['spatial_Z']:
        block_size = M 

    q_cvi = MeanFieldConjugateGaussian([
        legogp.approximate_posteriors.ConjugateGaussian(
            X=Z,
            block_size=block_size,
            surrogate_model = lambda X, Y, likelihood:  lego.models.GP(X, Y, prior=lego.transforms.Independent([lego.models.GP(X, kernel=K[q])]), likelihood=likelihood) # batch gp surrogate model 
        )
        for q in range(Q)
    ])


    if cmd_args['cvi']:
        print('--------- CVI --------')
        q = q_cvi

    elif cmd_args['vi']:
        print('--------- VI --------')
        # Init as CVI
        q_mf = [ q_cvi.approx_posteriors[q].surrogate.predict_f(Z, diagonal=False) for q in range(Q)]

        q = MeanFieldApproximatePosterior(
            approximate_posteriors = [
                legogp.approximate_posteriors.GaussianApproximatePosterior(
                    m = q_mf[q][0][:, None],
                    S = add_jitter(q_mf[q][1], 1e-5)
                )
                for q in range(Q)
            ]
        )

elif cmd_args['fp']:
    print('--------- FP --------')


    if cmd_args['cvi']:
        print('--------- CVI --------')

    legogp.settings.ng_jitter = 1e-5
    if cmd_args['no_Z']:

        block_size = Q

        q = FullConjugateGaussian(
            X = Z_all,
            num_latents=Q,
            block_size=block_size,
            surrogate_model = lambda X, Y, likelihood:  lego.models.GP(X, Y, likelihood=likelihood, prior=DataLatentPermutation(prior)), # batch gp surrogate model
        )
        #print(q.surrogate.get_objective())
        #print(q.surrogate.predict_blocks(Z_all, 1, Q, diagonal=False))

    elif cmd_args['spatial_Z']:

        raise NotImplementedError()
        block_size = M

        q = FullConjugateGaussian(
            X = Z_all,
            num_latents=Q,
            block_size=block_size,
            surrogate_model = lambda X, Y, likelihood:  lego.models.GP(X, Y, likelihood=likelihood, prior=DataLatentPermutation(prior)), # batch gp surrogate model
        )
        #print(q.surrogate.get_objective())
        #print(q.surrogate.predict_blocks(Z_all, 1, Q, diagonal=False))

    elif cmd_args['dense_Z']:

        block_size =  M * Q

        q = FullConjugateGaussian(
            X = Z_all,
            num_latents=Q,
            block_size=block_size,
            surrogate_model = lambda X, Y, likelihood:  lego.models.GP(X, Y, likelihood=likelihood, prior=DataLatentPermutation(prior)), # batch gp surrogate model
        )

        #print(q.surrogate.get_objective())
        #print(q.surrogate.predict_blocks(Z_all, M, M * Q, diagonal=False))
        #print(q.surrogate.predict_blocks(Z_all, 1, Q, diagonal=False))

    if cmd_args['vi']:
        print('--------- VI --------')
        legogp.settings.ng_jitter = 1e-6

        if True:
            # Initialise the same as CVI

            # Construct CVI
            q = FullConjugateGaussian(
                X = Z_all,
                num_latents=Q,
                block_size=block_size,
                surrogate_model = lambda X, Y, likelihood:  lego.models.GP(X, Y, likelihood=likelihood, prior=DataLatentPermutation(prior)), # batch gp surrogate model
            )
            # Get q(f)
            q_m, q_S = q.surrogate.predict_blocks(Z_all, M, M*Q, diagonal=False)
            q_m = q_m[..., None][0]
            q_S = q_S[0]
            q_m = q.surrogate.prior.unpermute_vec(q_m)
            q_S = q.surrogate.prior.unpermute_mat(q_S)
            q = lego.approximate_posteriors.FullGaussianApproximatePosterior(m = q_m, S= q_S+1e-5*np.eye(q_S.shape[0]))
        else:
            q = lego.approximate_posteriors.FullGaussianApproximatePosterior(dim=Q * M)



m = lego.models.GP(
    data=data,
    prior=prior,
    inference='Variational',
    whiten=False,
    minibatch_size=None,
    likelihood = [lego.likelihood.Gaussian(0.1) for p in range(P)],
    approximate_posterior = q,
    ell_samples=1000,
    prediction_samples=1000
)

#print(q_cvi.approx_posteriors[0].surrogate.get_objective())

print('OBJ: ', m.get_objective())
breakpoint()
#breakpoint()


if True:
    if True:
        #TODO: implement sequential natgrad trainer
        natgrad_trainer = NatGradTrainer(m, schedule='linear')
        natgrad_trainer.train([0.01, 0.1], 5)
        natgrad_trainer.train([1.0, 1.0], 1)
    else:
        train_adam(m, 500)

    print('OBJ: ', m.get_objective())

if cmd_args['cvi'] and cmd_args['fp']:
    XS_tiled = np.tile(XS, [Q, 1, 1])
    pred_mu, pred_var = m.predict_f(XS_tiled, squeeze=False)
else:
    pred_mu, pred_var = m.predict_f(XS, squeeze=False)

print(np.sum(pred_mu), np.sum(pred_var))

fig = plt.figure()

for p in range(P):
    if False:
        plt.fill_between(
            np.squeeze(XS),
            np.squeeze(pred_mu[p]) + 2*np.squeeze(np.sqrt(pred_var[p])),
            np.squeeze(pred_mu[p]) - 2*np.squeeze(np.sqrt(pred_var[p])),
            alpha = 0.4
        )
    plt.plot(XS, pred_mu[p])
    plt.scatter(X, Y[:, p], c='black')

    plt.scatter(Z, np.zeros(Z.shape[0]), c='grey')
plt.show()
