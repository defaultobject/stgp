import gpjax
from gpjax.model import GP
from gpjax.transform.multi_output import LMC, GPRN
import numpy as np
import jax.numpy as jnp
import objax
import matplotlib.pyplot as plt

from jax.config import config

from timeit import default_timer as timer

config.update('jax_disable_jit', False)
config.update("jax_enable_x64", True)


def run():
    Q = 5
    P = 5
    X = np.linspace(0, 1, 100)[:, None]
    #X = np.concatenate([X, X], axis=1)
    _Y = np.sin(X[:, 0]*10)[:, None]
    Y = np.concatenate([_Y for i in range(P)], axis=1)

    multi_task = True
    #m = GP(X, Y, inference='Variational', likelihood=[gpjax.likelihood.Poisson(binsize=0.1) for q in range(Q)], whiten=True)
    #m = GP(X, Y, inference='Variational', whiten=True)

    if multi_task:
        latents = [GP(X) for q in range(Q)]
        p = LMC(latents, output_dim=P)

        #m = GP(X, Y, likelihood=[gpjax.likelihood.Poisson(binsize=0.1) for q in range(P)], prior=p, inference='Variational', whiten=True)
        m = GP(X, Y, likelihood=[gpjax.likelihood.Poisson(binsize=0.1) for q in range(P)], inference='Variational', whiten=True)

    train_vars = m.vars()

    lr_adam = 0.01
    opt = objax.optimizer.Adam(train_vars)

    objective_fn = objax.Jit(m.get_objective, train_vars)
    grad_fn = objax.Jit(objax.GradValues(objective_fn, m.vars()), train_vars)

    start = timer()

    for i in range(100):
        grad, val = grad_fn()
        if i % 10 == 0:
            print(val)
        opt(lr_adam, grad)

    end = timer()
    training_time = end - start
    print('training_time: ', training_time)

    pred_fn = objax.Jit(m.predict, m.vars())
    mean, var = pred_fn(X)

    if Q > 1:
        mean = mean[0]
        var = var[0]

    plt.fill_between(np.squeeze(X), mean-2*np.sqrt(var), mean+2*np.sqrt(var), alpha=0.3)
    plt.scatter(X, _Y)
    plt.plot(X, mean)
    plt.show()


gpjax.settings.force_black_box = False
gpjax.settings.jitter = 1e-6

if False:
    print('No loops')
    run()
else:
    print('With loops')

    with gpjax.settings.use_loops():
        run()


