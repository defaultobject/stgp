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
    Q = 1
    X = np.linspace(0, 1, 100)[:, None]
    #X = np.concatenate([X, X], axis=1)
    _Y = np.sin(X[:, 0]*10)[:, None]
    Y = np.concatenate([_Y for i in range(Q)], axis=1)

    #m = GP(X, Y, inference='Variational', likelihood=gpjax.likelihood.Poisson(binsize=0.1), whiten=True)
    m = GP(X, Y, inference='Variational', whiten=True)

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

    plt.fill_between(np.squeeze(X), mean-2*np.sqrt(var), mean+2*np.sqrt(var), alpha=0.3)
    plt.scatter(X, _Y)
    plt.plot(X, mean)
    plt.show()


if False:
    print('No loops')
    run()
else:
    print('With loops')
    gpjax.settings.force_black_box = True
    gpjax.settings.jitter = 1e-6
    with gpjax.settings.use_loops():
        run()


