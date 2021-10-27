import jax
import objax
import numpy as np

import gpax
from gpax.trainers import SimpleTrainer
from gpax.trainers.callbacks import progress_bar_callback
from gpax.kernels.deep_kernels import DeepRBF
from gpax.kernels import RBF
from gpax.models import GP

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.set_printoptions(linewidth=200)

np.random.seed(0)


N = 100

X = np.linspace(0, 1, N)[:, None]
XS = np.linspace(-0.5, 1.5, 1000)[:, None]

Y = np.sin(10*X) + 0.1*np.random.randn(X.shape[0])[:, None]

P = 2
Y = np.hstack([Y*(i+1) for i in range(P)])

print(X.shape, Y.shape)

if False:
    m = gpax.models.GP(X=X, Y = Y, kernel=[DeepRBF(RBF(input_dim=1), input_dim=1) for p in range(P)], inference='Batch')
else:
    Q = 3
    latents = [
        GP(X=X, kernel=RBF(input_dim=1), latent=True)
        for q in range(Q)
    ]
    prior = gpax.transforms.multi_output.LMC_Unit_Tri(latents, output_dim = P)
    m = GP(X=X, Y = Y, prior=prior, inference='Batch')

epochs = 200

m.get_objective()

if True:
    callback = progress_bar_callback(epochs)

    learning_curve, training_time = SimpleTrainer().train(
        m, 
        objax.optimizer.Adam,
        0.01,
        epochs,
        callback = callback
    )

if False:
    plt.plot(learning_curve)
    plt.show()

mu, var = m.predict(XS)

colors = cm.rainbow(np.linspace(0, 1, P))

for p in range(P):
    plt.fill_between(
        np.squeeze(XS), 
        np.squeeze(mu[p]) - 1.96*np.sqrt(np.squeeze(var[p])), 
        np.squeeze(mu[p]) + 1.96*np.sqrt(np.squeeze(var[p])), 
        alpha=0.4,
        facecolor = colors[p],
        label=f'Output {p}'
    )
    plt.plot(XS, mu[p], c=colors[p])
    plt.scatter(X, Y[:, p], c='black')

plt.legend()
plt.show()
