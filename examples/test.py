import gpax
from gpax.model import GP
from gpax.transform.multi_output import LMC, GPRN
from gpax.sparsity import NoSparsity
from gpax.kernel import RBF
from gpax.trainer import SimpleTrainer
from gpax.trainer.callbacks import progress_bar_callback
import numpy as np
import jax.numpy as jnp
import jax
import objax
import matplotlib.pyplot as plt
import json
import os
#os.environ['XLA_FLAGS']='--xla_force_host_platform_device_count=4'

from jax.config import config

from timeit import default_timer as timer

JIT = True

config.update('jax_disable_jit', False)
config.update("jax_enable_x64", True)


def run():
    train_model = True
    checkpoint = True
    Q = 1
    P = 1
    X = np.linspace(0, 1, 100)[:, None]
    num_epochs = 500
    #X = np.concatenate([X, X], axis=1)
    _Y = (np.sin(X[:, 0]*10)+0.1*np.random.randn(100))[:, None] 
    Y = np.concatenate([_Y for i in range(P)], axis=1)

    model = 'lmc'
    #m = GP(X, Y, inference='Variational', likelihood=[gpax.likelihood.Poisson(binsize=0.1) for q in range(Q)], whiten=True)
    #m = GP(X, Y, inference='Variational', whiten=True)

    if model == 'lmc':
        print('running LMC')
        latents = [GP(X, kernel=RBF(lengthscales=[0.1])) for q in range(Q)]
        prior = LMC(latents, output_dim=P)


        #m = GP(X, Y, likelihood=[gpax.likelihood.Poisson(binsize=0.1) for q in range(P)], prior=p, inference='Variational', whiten=True)
        m = GP(X, Y, likelihood=[gpax.likelihood.Gaussian(variance=0.1) for q in range(P)], prior=prior, inference='Variational', whiten=False)

        #m = GP(X, Y, likelihood=[gpax.likelihood.Poisson(binsize=0.1) for q in range(P)], inference='Variational', whiten=True)
    elif model == 'gprn':
        print('running GPRN')
        latents_f = [GP(X, kernel=RBF(lengthscales=[0.1])) for q in range(Q)]
        latents_W = [[GP(X, kernel=RBF(lengthscales=[0.1])) for q in range(Q)] for p in range(P)]

        prior = GPRN(latents_f, latents_W)

        m = GP(X, Y, likelihood=[gpax.likelihood.Gaussian() for q in range(P)], prior=prior, inference='Variational', whiten=False)

    class Model(objax.Module):
        def __init__(self, Y):
            self.Y = jnp.array(Y)
            self.w = objax.StateVar(jnp.array([1.0]))
            self.f = objax.TrainVar(jnp.ones_like(Y))

        def get_objective(self):
            return  jnp.sum(jnp.square(self.Y - self.f.value*self.w.value))

        def predict(self, XS):
            return self.f.value*self.w.value, self.f.value*self.w.value

    #m = Model(Y)

    if train_model:
        train_vars = m.vars()
        print(json.dumps({str(a): ''  for a in m.vars().keys()}, indent=3))

        if JIT:
            callback = progress_bar_callback(num_epochs)
        else:
            callback = None

        learning_curve, training_time = SimpleTrainer().train(
            m, 
            objax.optimizer.Adam,
            0.01,
            num_epochs,
            callback = callback
        )

        print('training_time: ', training_time)


        if checkpoint:
            m.checkpoint()
    else:
        learning_curve = None
        m.load_from_checkpoint()

    if model == 'lmc':

        pass

    if learning_curve is not None:
        print(np.array(learning_curve))
        plt.plot(learning_curve)
        plt.show()


    #pred_fn = objax.Jit(m.predict, m.vars())
    pred_fn = m.predict
    mean, var = pred_fn(X)

    mean = mean.reshape([P, X.shape[0]])
    var = var.reshape([P, X.shape[0]])

    if True:


        for p in range(P):
            plt.fill_between(np.squeeze(X), mean[p]-2*np.sqrt(var[p]), mean[p]+2*np.sqrt(var[p]), alpha=0.3)
            plt.scatter(X, Y[:, p], label=p)
            plt.plot(X, mean[p], label=p)

    plt.show()


gpax.settings.force_black_box = False
gpax.settings.jitter = 1e-8

#jax.profiler.start_trace("/tmp/tensorboard")

if False:
    print('No loops')
    run()
else:
    print('With loops')

    with gpax.settings.use_loops():
        run()

#jax.profiler.stop_trace()

