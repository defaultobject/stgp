"""Batch Gaussian Process Regression with a Derivative Observations"""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_disable_jit", False)


import numpy as np

import stgp
from stgp.trainers import ScipyTrainer
from stgp.trainers.jaxopt import JaxoptTrainer
import jaxopt
import optax

from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import ApproxSDEPeriodic, Periodic
from stgp.kernels.periodic import ApproxSDEPeriodic_BN
from stgp.transforms import Independent
from stgp.transforms.sdes import LTI_SDE
from stgp.likelihood import (
    Gaussian,
    DiagonalGaussian,
    ReshapedGaussian,
    ProductLikelihood,
)
from stgp.models import GP
from stgp.trainers.standard import ADAM

import matplotlib.pyplot as plt

import pickle

include_dt = False
one_output = True
custom = False
n_terms = 8
state_space = True
train = True
use_BN = True


i = 0


if False:
    kernel = ApproxSDEPeriodic(
        0.3896384,
        3.78083853,
        2.02294663,
        n_terms=n_terms,
        include_dt=include_dt,
        use_custom_bessel_ive=custom,
    )
    kernel_bn = ApproxSDEPeriodic_BN(
        0.3896384,
        3.78083853,
        2.02294663,
        n_terms=n_terms,
        include_dt=include_dt,
        use_custom_bessel_ive=custom,
    )

    F, L, Qc, H, Pinf = kernel.to_ss(None)
    F_bn, L_bn, Qc_bn, H_bn, Pinf_bn = kernel_bn.to_ss(None)

    print(np.sum(kernel.state_size() - kernel_bn.state_size()))

    print(F.shape, F_bn.shape, np.sum(F - F_bn))
    print(L.shape, L_bn.shape, np.sum(L - L_bn))
    print(Qc.shape, Qc_bn.shape, np.sum(Qc - Qc_bn))
    print(H.shape, H_bn.shape, np.sum(H - H_bn))
    print(Pinf.shape, Pinf_bn.shape, np.sum(Pinf - Pinf_bn))
    print(np.sum(kernel.expm(0.01) - kernel_bn.expm(0.01)))
    breakpoint()

if one_output:
    Q = 1
else:
    Q = 2


if True:
    data = pickle.load(
        open(
            "/Users/oliverhamelijnck/Documents/projects/pigp/code/experiments/lotka_volterra/data/train_data_0.pickle",
            "rb",
        )
    )
    X = data["train"]["X"]
    Y = data["train"]["Y"]
    np.random.seed(0)

    if False:
        import pandas as pd

        df = pd.DataFrame({"X": np.squeeze(X), "Y": np.squeeze(Y[:, 0])})
        df.to_csv("~/Downloads/lotka.csv")

    if one_output:
        if include_dt:
            Y = np.vstack([Y[:, i], Y[:, i] * np.NaN]).T
        else:
            Y = Y[:, [i]]
    else:
        if include_dt:
            Y = np.vstack([Y[:, 0], Y[:, 0] * np.NaN, Y[:, 1], Y[:, 1] * np.NaN]).T

    print(Y.shape)
    Y_all = Y

    XS = np.linspace(-50, 100, 1000)[:, None]
    N = X.shape[0]

else:
    # Construct data
    f = lambda x: np.cos(10 * x) + np.cos(5 * x) + np.cos(x)

    N = 100
    x = np.linspace(0, 2, N)
    y = f(x) + 0.01 * np.random.rand(N)

    X = x[:, None]
    Y_all = y[:, None]

    # remove non derivative observations
    Y = np.copy(Y_all)

    # testing locations
    XS = np.linspace(-10, 10, 1000)[:, None]

if False:
    plt.plot(X, Y_all)
    plt.show()


if use_BN:
    kern = ApproxSDEPeriodic_BN
else:
    kern = ApproxSDEPeriodic

if state_space:
    # construct model kernel and likelihood
    # base_kernel_1d = ScaledMatern32(input_dim = 1, lengthscales = [0.1], variance=1.0)
    # base_kernel_1d = ApproxSDEPeriodic(0.38922899, 1.58582818, 0.59429964, n_terms=3, include_dt=include_dt, use_custom_bessel_ive=False)
    # frequency, lengthscale, variance
    if i == 0:
        # kernel = kern(0.38811869, 0.93682507, 3.18527886, n_terms=n_terms, include_dt=include_dt, use_custom_bessel_ive=custom)
        # kernel = kern(0.1, 0.1, 1.0, n_terms=n_terms, include_dt=include_dt, use_custom_bessel_ive=custom)
        kernel = kern(
            1.0,
            1.0,
            1.0,
            n_terms=n_terms,
            include_dt=include_dt,
            use_custom_bessel_ive=custom,
        )
    else:
        kernel = kern(
            0.38954308,
            0.57183053,
            3.12030884,
            n_terms=n_terms,
            include_dt=include_dt,
            use_custom_bessel_ive=custom,
        )

    latent_gp = [
        GP(
            sparsity=stgp.sparsity.NoSparsity(Z=X),
            # kernel = ApproxSDEPeriodic(0.3896384, 3.78083853, 2.02294663, n_terms=n_terms, include_dt=include_dt, use_custom_bessel_ive=custom)
            kernel=kernel,
        )
        for q in range(Q)
    ]

    latent_gp = LTI_SDE(Independent(latent_gp))

    if include_dt:
        lik = ReshapedGaussian(DiagonalGaussian(variance=[0.1] * (2 * Q)), N, 2 * Q)
    else:
        if Q == 1:
            lik = ReshapedGaussian(Gaussian(variance=0.1), N, Q)
        else:
            lik = ReshapedGaussian(DiagonalGaussian(variance=[0.1] * Q), N, Q)

    # Create Model
    m = stgp.models.GP(
        data=stgp.data.MultiOutputTemporalData(X, Y, sort=True),
        prior=latent_gp,
        likelihood=lik,
        full_state_observed=False,
        inference="Sequential",
    )
else:
    m = stgp.models.GP(
        data=stgp.data.Data(X, Y),
        prior=Independent(
            [
                GP(
                    sparsity=stgp.sparsity.NoSparsity(Z=X),
                    kernel=Periodic(0.39004889, 0.52517587, 1.80682043),
                )
            ]
        ),
        likelihood=ProductLikelihood([Gaussian(variance=0.1)]),
    )
m.print()
m.get_objective()

# train
if train:
    max_iters = 20
    print(m.get_objective())
    if True:
        adam = True
        if not adam:
            trainer = ScipyTrainer(m, "L-BFGS-B")
        else:
            trainer = ADAM(m)

        if False:
            lc_arr = []
            for i in range(max_iters):
                _lc, _ = trainer.train(0.01, 1)
                lc_arr.append(np.squeeze(_lc))
        else:
            lc_arr, _ = trainer.train(
                0.01, max_iters, callback=progress_bar_callback(max_iters)
            )
    else:
        if False:
            maxiter = 20
            max_iters = 1
        else:
            maxiter = 1
            max_iters = 20

        trainer = JaxoptTrainer(
            m,
            lambda *args, **kwargs: jaxopt.OptaxSolver(
                opt=optax.adam(0.01), *args, **kwargs
            ),
            maxiter=maxiter,
        )
        # trainer = JaxoptTrainer(m, lambda *args, **kwargs: jaxopt.LBFGS(*args, **kwargs, maxls=20), maxiter=maxiter)
        # trainer = JaxoptTrainer(m, lambda *args, **kwargs: jaxopt.LBFGS(*args, **kwargs, maxls=20), maxiter=maxiter)
        # trainer = JaxoptTrainer(m, lambda *args, **kwargs: jaxopt.ScipyMinimize(*args, **kwargs, method='l-bfgs-b'), maxiter=1)
        lc_arr, _ = trainer.train(
            None,
            max_iters,
            callback=progress_bar_callback(max_iters),
            state_has_obj=False,
        )
    plt.plot(lc_arr)
    plt.show()
    print(m.get_objective())
    m.print()
    # m.checkpoint('cos_gp')
else:
    print(m.get_objective())
    # m.load_from_checkpoint('cos_gp')

# predict
pred_mu, pred_var = m.predict_f(XS, filter_only=False)

pred_mu = np.squeeze(pred_mu)
pred_var = np.squeeze(pred_var)


if len(pred_mu.shape) == 1:
    pred_mu = pred_mu[:, None]
    pred_var = pred_var[:, None]

# plot
if include_dt:
    D = Q * 2
else:
    D = Q

fig, axes = plt.subplots(D)

if D == 1:
    axes = [axes]

for d in range(D):
    axes[d].fill_between(
        XS[:, 0],
        pred_mu[:, d] - 1.96 * np.sqrt(pred_var[:, d]),
        pred_mu[:, d] + 1.96 * np.sqrt(pred_var[:, d]),
        alpha=0.4,
    )

    axes[d].plot(XS, pred_mu[:, d])

    axes[d].scatter(X, Y_all[:, d], c="grey")

    axes[d].scatter(X, Y[:, d], c="black")

if include_dt:
    # plot numerical derivative
    axes[1].plot(XS, np.gradient(pred_mu[:, 0], XS[:, 0]), linestyle="--")
    if not one_output:
        axes[3].plot(XS, np.gradient(pred_mu[:, 2], XS[:, 0]), linestyle="--")


plt.show()
