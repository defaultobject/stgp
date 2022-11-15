""" Conjugate Variational Gaussian Process Regression with a Spatio-Temporal State-Space Surrogate Model"""


import sys
sys.path.append('../..')

import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
import objax
import numpy as np

from example_utils.data_zoo import single_output_spatial_data
from example_utils import colors

import stgp
from stgp.models import GP
from stgp.trainers import ScipyTrainer, GradDescentTrainer, NatGradTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import SpatioTemporalSeperableKernel, Matern32, RBF 
from stgp.data import SpatioTemporalData
from stgp.likelihood import Gaussian, ReshapedGaussian
from stgp.transforms.sdes import LTI_SDE
from stgp.transforms import Independent
from stgp.approximate_posteriors import MeanFieldConjugateGaussian, ConjugateGaussian

from tqdm import trange
import matplotlib.pyplot as plt

# Construct Data
XS, X, Y = single_output_spatial_data(20, 20, 200, 200, seed=0)

# Construct Model
Q = 1
st_data = SpatioTemporalData(X=X, Y=Y, sort=True)
sparsity = stgp.sparsity.NoSparsity(Z = st_data.X)

# Setup Prior Model
kern= SpatioTemporalSeperableKernel(
    Matern32(input_dim=1, lengthscales=[0.1], active_dims=[0]),
    RBF(input_dim=1, lengthscales=[0.1], active_dims=[1])
)

latent_gp = GP(
    sparsity = sparsity, 
    kernel = kern,
    prior = True
)
prior = Independent([latent_gp])

# Setup Surrogate Model
approximate_posterior = MeanFieldConjugateGaussian([
    ConjugateGaussian(
        X=st_data._X,
        block_size = st_data.Ns,
        num_blocks = st_data.Nt,
        num_latents = 1,
        surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
            data=SpatioTemporalData(X=X, Y=np.reshape(Y, [st_data.Nt, 1, st_data.Ns]), sort=False), # Data should already be in the correct format
            prior=LTI_SDE(Independent([latent_gp])), 
            likelihood=[likelihood[q]],
            inference='Sequential'
        )  
    )
    for q in range(Q)
])

m = GP(
    data = st_data,
    prior = prior,
    likelihood = Gaussian(0.1), 
    inference='Variational',
    approximate_posterior = approximate_posterior
)

#print(m.get_objective())
#print(m.predict_f(X))


# Train
if True:
    max_iters = 100

    ng_trainer = NatGradTrainer(m)
    m.approximate_posterior.fix()

    trainer = GradDescentTrainer(m, objax.optimizer.Adam)

    lc, _ = ng_trainer.train(1.0, 1)
    lc_arr = [lc]
    for i in trange(max_iters):
        trainer.train(0.01, 1)
        lc_i, _ = ng_trainer.train(1.0, 1)
        lc_arr.append(lc_i)

    plt.plot(lc_arr)
    plt.show()

# Predict
pred_mu, pred_var = m.predict_f(XS)

# Plot
fig, axes = plt.subplots(1, 2)

axes[0].set_title('Mean')
axes[0].imshow(pred_mu.reshape(200, 200))

axes[1].set_title('Variance')
axes[1].imshow(pred_var.reshape(200, 200))

plt.show()

