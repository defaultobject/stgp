import numpy as np
import scipy as sp
import gpflow
from gpflow.utilities import print_summary

import tensorflow as tf

import matplotlib.pyplot as plt

np.random.seed(0)

def nonstationary_function(x, seed=0):
    # See https://www.ics.uci.edu/~pazzani/Publications/ssdb99.pdf
    #   or https://archive.ics.uci.edu/ml/datasets/Pseudo+Periodic+Synthetic+Time+Series
    f = np.zeros(x.shape[0])
    np.random.seed(seed)
    for i in range(3, 7):
        rand_i = np.random.uniform(low=0.0, high=np.power(2, i), size=(1,))
        _f = (1/np.power(2, i)) * np.sin( 2*np.pi*( np.power(2,2+i)+rand_i) * x * 3.0)
        f += _f
    return f

x = np.linspace(0, 1, 100)
X = x[:, None]

u1 = nonstationary_function(10*x, seed=0)[:, None]
u2 = nonstationary_function(x, seed=2)[:, None]

W = np.array([[2, 0.6], [0.6, 1.0]])

Y_all = (W @ np.hstack([u1, u2]).T).T

missing_region = [40, 60]

Y = Y_all.copy()
Y[missing_region[0]:missing_region[1], 0] = np.NaN

P = 1
Y_all = Y_all[:, 1][:, None]
Y = Y[:, 1][:, None]

XS = np.linspace(0, 1, 1000)[:, None]


#model
k = gpflow.kernels.RBF(lengthscales=0.1)
m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
m.likelihood.variance.assign(1.0)

def run_adam(model, iterations):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []

    data = (X, Y)

    training_loss = m.training_loss_closure()

    optimizer = tf.optimizers.Adam(0.01)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
            print(elbo)
    return logf

run_adam(m, 1000)
print_summary(m)

if False:
    K_xs = k.K(XS, XS)
    K_xs_x = k.K(XS, X)
    K_xx = k.K(X, X) + m.likelihood.variance*np.eye(X.shape[0])
    K_xx_chol = sp.linalg.cholesky(K_xx)
    var = K_xs - K_xs_x @ sp.linalg.cho_solve(
        (K_xx_chol, True),
        K_xs_x.T
    )

    breakpoint()

mu, var = m.predict_f(XS)

print(var)

plt.fill_between(
    np.squeeze(XS), 
    np.squeeze(mu) - 2 * np.sqrt(np.squeeze(var)),
    np.squeeze(mu) + 2 * np.sqrt(np.squeeze(var)),
    alpha=0.3
)
plt.plot(XS, mu)
plt.scatter(X, Y)
plt.show()
