""" Moment Mathched Batched Gaussian Process Regression """
import sys
sys.path.append('../')
sys.path.append('../../')

import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
import objax
import numpy as np

from example_utils.data_zoo import single_output_timeseries
from example_utils import colors
from stgp.trainers import ScipyTrainer, GradDescentTrainer
from stgp.trainers.callbacks import progress_bar_callback

import stgp
from stgp.models import GP
from stgp.transforms import Independent
from stgp.data import Data
from stgp.sparsity import NoSparsity, FullSparsity
from stgp.kernels import ScaleKernel, RBF
from stgp.likelihood import Gaussian
from stgp.kernels.deep_kernels import DeepRBF
from stgp.models.wrappers import MultiObjectiveModel
from stgp.trainers.standard import VB_NG_ADAM

import matplotlib.pyplot as plt

# TODO: to try
#stgp.settings.use_loop_mode = True

# Construct Data
XS, X, Y_lr = single_output_timeseries(100, 1000, seed=0)
Y_hr = np.exp(Y_lr)
Y_hr_all =  np.copy(Y_hr)

# remove part of high fidelity for testing
Y_hr[10:30, :] = np.NaN


# construct model

N, D = X.shape
_, P = Y_lr.shape

Z = X[:10]


m_lr = GP(
    data = Data(X, Y_lr),
    prior = Independent([
        GP(
            sparsity = FullSparsity(Z=Z), 
            kernel = ScaleKernel(RBF(input_dim=D, lengthscales=[0.1 for d in range(D)]))
        )
    ]),
    likelihood = [Gaussian(variance=0.1)],
    inference='Variational'
)

m_hr = GP(
    data = Data(X, Y_hr),
    prior = Independent([
        GP(
            sparsity = FullSparsity(Z=Z), 
            kernel = ScaleKernel(DeepRBF(parent=m_lr, lengthscale=[1.0]))
        )
    ]),
    likelihood = [Gaussian(variance=0.1)],
    inference='Variational'
)

#train

# create multi objective model so that the objective is
#  m_hr.get_objective() + m_lr.get_objective()

m = MultiObjectiveModel([m_hr, m_lr])

# train m_lr first


if True:
    lr_lr, _ = VB_NG_ADAM(m_lr).train([0.01, 0.9], [200, [1, 1]], callback=progress_bar_callback(200))
    lr_hr, _ = VB_NG_ADAM(m_hr).train([0.01, 0.9], [200, [1, 1]], callback=progress_bar_callback(200))

    # Train
    epochs = 200
    callback = progress_bar_callback(epochs)
    learning_curve, training_time = GradDescentTrainer(
        m, 
        objax.optimizer.Adam
    ).train(
        0.01,
        epochs,
        callback = callback
    )

    # Plot learning curve
    plt.plot(learning_curve)
    plt.show()


# predict
pred_mu_lr, pred_var_lr = m_lr.predict_f(XS)
pred_mu_hr, pred_var_hr = m_hr.predict_f(XS)

fig, axes = plt.subplots(2, 1)

#plot low fidelity
axes[0].fill_between(
    np.squeeze(XS), 
    np.squeeze(pred_mu_lr - 2* np.sqrt(pred_var_lr)),
    np.squeeze(pred_mu_lr + 2* np.sqrt(pred_var_lr)),
    alpha = 0.4
)
axes[0].plot(XS, pred_mu_lr)
axes[0].scatter(X, Y_lr)


#plot high fidelity
axes[1].fill_between(
    np.squeeze(XS), 
    np.squeeze(pred_mu_hr - 2* np.sqrt(pred_var_hr)),
    np.squeeze(pred_mu_hr + 2* np.sqrt(pred_var_hr)),
    alpha = 0.4
)
axes[1].plot(XS, pred_mu_hr)
axes[1].scatter(X, Y_hr_all, c='grey')
axes[1].scatter(X, Y_hr)
plt.show()



