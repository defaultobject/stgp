import sys
sys.path.append('../../')

import gpflow
from gpflow import set_trainable
from gpflow.ci_utils import reduce_in_tests
from gpflow.models import GPR, SGPR, SVGP, VGP
from gpflow.optimizers import NaturalGradient
from gpflow.optimizers.natgrad import XiSqrtMeanVar

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

def single_output_timeseries(N, NS, seed=0):
    np.random.seed(seed)

    x = np.linspace(0, 1, N)
    y = np.sin(x*10) + 0.1*np.random.randn(N)
    X = x[:, None]
    Y = y[:, None]

    XS = np.linspace(-1, 2, 1000)[:, None]

    return XS, X, Y

XS, X, Y = single_output_timeseries(100, 1000, seed=0)

N = X.shape[0]
M = 10
Z = np.linspace(np.min(X), np.max(X), M)[:, None]
batch_size=10

svgp = SVGP(
    kernel=gpflow.kernels.RBF(lengthscales=0.1),
    likelihood=gpflow.likelihoods.Gaussian(0.1),
    inducing_variable=Z,
    num_data = N
)


# Stop Adam from optimizing the variational parameters
set_trainable(svgp.q_mu, False)
set_trainable(svgp.q_sqrt, False)

elbo = tf.function(svgp.elbo)


data = (X, Y)

#print(svgp.elbo(data).numpy())
#breakpoint()

data_minibatch = (
    tf.data.Dataset.from_tensor_slices(data)
    .repeat()
    .shuffle(N)
    .batch(batch_size)
)
data_minibatch_it = iter(data_minibatch)

if False:
    import itertools
    batches = [
        minibatch[0] for minibatch in itertools.islice(data_minibatch_it, 100)
    ]

    print([np.unique(d).shape[0] for d in batches])
    breakpoint()
    evals = [
        elbo(minibatch).numpy() for minibatch in itertools.islice(data_minibatch_it, 1000)
    ]


    plt.hist(evals)
    plt.show()

    breakpoint()


svgp_objective = svgp.training_loss_closure(data_minibatch_it, compile=True)

optimizer = tf.optimizers.Adam(0.01)
natgrad_opt = NaturalGradient(gamma=0.99)

lc_arr = []
variational_params = [(svgp.q_mu, svgp.q_sqrt)]

for _ in range(100):
    natgrad_opt.minimize(svgp_objective, var_list=variational_params)
    #print(svgp_objective().numpy())
    #breakpoint()
    optimizer.minimize(svgp_objective, var_list=svgp.trainable_variables)
    lc_arr.append(svgp_objective().numpy())

plt.plot(lc_arr)
plt.show()


pred_mu, pred_var = svgp.predict_y(XS)


plt.fill_between(
    np.squeeze(XS), 
    np.squeeze(pred_mu - 1.96*np.sqrt(pred_var)), 
    np.squeeze(pred_mu + 1.96*np.sqrt(pred_var)), 
    facecolor='blue', 
    alpha=0.3
)
plt.plot(XS, pred_mu, color='blue', label='GP Fit')
plt.scatter(X, Y, color='grey', label='Training Data')

plt.legend()
plt.show()


