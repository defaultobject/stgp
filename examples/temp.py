import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)
import jax.numpy as jnp

import objax
import objax.zoo
import objax.zoo.dnnet
import numpy as np
import pandas as pd

import gpax
from gpax.trainers import SimpleTrainer, ScipyTrainer
from gpax.trainers.callbacks import progress_bar_callback
from gpax.kernels.deep_kernels import DeepRBF, DeepHetreo, DeepNN
from gpax.kernels import WhiteNoiseKernel
from gpax.kernels import RBF
from gpax.models import GP

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pathlib import Path

np.set_printoptions(linewidth=200)

np.random.seed(0)


data_path = '/Users/ohamelijnck/Documents/projects/deep_kernel/code/experiments/_motorcycle/data/motorcycle_data.txt'

f = open(data_path, "r")
tmp = f.read().split('\n')
data = []
for i in tmp:
    tmp2 = i.split('\t')
    data.append(tmp2)
data = np.array(data,dtype = float)

X = data[:,0][:, None]
Y = data[:,1][:, None]

Y = Y-np.mean(Y)
Y = Y/np.std(Y)

XS = np.linspace(np.min(X[:, 0])-20, np.max(X[:, 0])+20, 500)[:, None]

if False:
    N = 300

    X = np.linspace(0, 1, N)[:, None]
    XS = np.linspace(-0.5, 1.5, 1000)[:, None]

    #Y = np.clip(np.sin(10*X), -0.8, 0.8) + 0.05*np.random.randn(X.shape[0])[:, None]

    x = X[:, 0]
    y = np.sin(4*x) + np.random.normal(size = X.shape[0], scale = abs(0.5-1*X[:,0]), loc = 0)
    Y = y[:, None]

if False:
    plt.scatter(X, Y)
    plt.show()

P = 3
Y = np.hstack([Y*(i+1) for i in range(P)])

Y_latent = pd.DataFrame(Y).rolling(10).std()
Y_latent = np.nan_to_num(Y_latent, nan=0.001)

prior_posterior = gpax.models.GP(X=X, Y=Y, kernel=[RBF(lengthscales=[1.0]) for p in range(P)])

prior_posterior.get_objective()

prior_p = gpax.transforms.Independent(
    latent = prior_posterior, 
    prior=False
)
prior_p = gpax.transforms.transform.DeepKernel_One2One(
    prior = prior_p,
    kernels = [DeepRBF(input_dim=1)]
)


latents = [
    GP(X=X, kernel=RBF(input_dim=1), latent=True)
    for q in range(P)
]

prior = gpax.transforms.multi_output.LMC_Unit_Tri(latents, output_dim = P)

prior = gpax.transforms.SumTransform(prior, prior_p)

#print(prior.covar(XS, X).shape)
#print(prior.var(XS).shape)
#print(prior.full_var(X).shape)
#print(prior.mean(X).shape)

m = gpax.models.GP(
    X=X, 
    Y = Y,  
    prior = prior, 
    inference='Batch', 
    likelihood=[gpax.likelihood.Gaussian(1.0) for p in range(P)]
)
m.get_objective()
breakpoint()
mu, var = m.predict(XS, diagonal=False)
print(mu.shape, var.shape)
breakpoint()

subsample = 10
latent_noise_gp = gpax.models.GP(X=X[::subsample, :], Y = Y_latent[::subsample, :], latent_y=True, kernel=[RBF(lengthscales=[10.0])], inference='Batch', likelihood=[gpax.likelihood.Gaussian(1.0)])
mu, var = latent_noise_gp.predict(X)


if True:
    #kern = [DeepRBF(RBF(input_dim=1)+WhiteNoiseKernel(), input_dim=1) for p in range(P)]
    #kern = [DeepRBF(DeepRBF(RBF(input_dim=1)+WhiteNoiseKernel())) for p in range(P)]
    #kern = [DeepRBF(DeepRBF(RBF(input_dim=1)+WhiteNoiseKernel())+WhiteNoiseKernel()) for p in range(P)]
    #kern = [RBF(lengthscales=[5.0]) for p in range(P)]
    #kern = [DeepRBF(RBF(lengthscales=[5.0])) for p in range(P)]
    #kern = [DeepRBF(DeepRBF(RBF(lengthscales=[5.0]))) for p in range(P)]
    if False:
        kern = [RBF(variance=0.1) for p in range(P)]
        K = DeepHetreo(latent_noise_gp)
        m = gpax.models.GP(X=X, Y = Y, kernel=kern, inference='Batch', likelihood=[gpax.likelihood.GaussianParameterised(K)])
    else:
        if False:
            kern = [RBF(variance=0.1)*DeepRBF(WhiteNoiseKernel()) for p in range(P)]
            m = gpax.models.GP(X=X, Y = Y, kernel=kern, inference='Batch', likelihood=[gpax.likelihood.Gaussian(variance=0.01)])
        else:
            if True:
                latent_noise_gp = gpax.models.GP(X=X[::subsample, :], Y = np.hstack([Y_latent[::subsample, :] for p in range(3)]), latent_y=True, inference='Batch')
                #kern = [RBF(variance=0.1)+DeepHetreo(latent_noise_gp) for p in range(P)]
                #kern = [RBF(variance=0.1)*DeepRBF(DeepHetreo(latent_noise_gp, ignore_var=True)) for p in range(P)]
                #kern = [RBF(variance=0.1)*DeepRBF(DeepHetreo(latent_noise_gp, ignore_var=False)) for p in range(P)]
                kern = [RBF(variance=0.1)*DeepRBF(parent_model=latent_noise_gp) for p in range(P)]
            else:
                class NN(objax.Module):
                    def __init__(self, nn):
                        self.nn = nn

                    def predict(self, X, *args, **kwargs):
                        res = self.nn(X)
                        mu = res[:, 0]
                        var = res[:, 1]
                        var = objax.functional.softplus(var) #ensure positivity
                        return mu, var

                kern = [RBF(variance=0.1)*DeepRBF(DeepHetreo(NN(
                    nn = objax.zoo.dnnet.DNNet(layer_sizes=[1, 100, 50, 2], activation=objax.functional.relu)
                ), ignore_var=False)) for p in range(P)]

            m = gpax.models.GP(X=X, Y = Y, kernel=kern, inference='Batch', likelihood=[gpax.likelihood.Gaussian(variance=0.01)])
else:
    Q = P
    latents = [
        GP(X=X, kernel=RBF(input_dim=1), latent=True)
        for q in range(Q)
    ]
    prior = gpax.transforms.multi_output.LMC_Corr(latents, output_dim = P)
    m = GP(X=X, Y = Y, prior=prior, inference='Batch')

epochs = 3000

m.get_objective()

print(m.vars())


if True:
    print(m.vars())

    callback = progress_bar_callback(epochs)
    learning_curve, training_time = SimpleTrainer().train(
        m, 
        objax.optimizer.Adam,
        0.01,
        epochs,
        callback = callback
    )


    if True:
        plt.plot(learning_curve)
        plt.show()


    m.checkpoint('temp_model')

if False:
    callback = progress_bar_callback(epochs)

    print(m.vars())
    learning_curve, training_time = ScipyTrainer().train(
        m, 
        'BFGS',
        0.01,
        epochs,
        callback = callback
    )
    m.checkpoint('temp_model')



if False:
    m.load_from_checkpoint('temp_model')


print(m.vars())

mu, var = m.predict(XS)

mu = np.reshape(mu, [P, XS.shape[0]])
var = np.reshape(var, [P, XS.shape[0]])

#breakpoint()
#var = var + np.reshape(np.diag(m.likelihood[0].variance(XS)), [P, XS.shape[0]])

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
