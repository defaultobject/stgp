"""Conjugate Variational Gaussian Process Regression on a temporal dataset"""

import sys

sys.path.append("../")
sys.path.append("../../")

from jax.config import config as jax_config

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_disable_jit", False)
import numpy as np

from example_utils.data_zoo import single_output_timeseries
from example_utils import colors
from stgp.trainers import NatGradTrainer
from stgp.kernels import ScaleKernel, Matern52, ScaledMatern52
from stgp.likelihood import Gaussian, GaussianProductLikelihood
from stgp.data import Data, TemporalData
from stgp.transforms import Independent
from stgp.transforms.sdes import LTI_SDE
from stgp.approximate_posteriors import (
    MeanFieldConjugateGaussian,
    ConjugateGaussian,
    ConjugatePrecisionGaussian,
)
from stgp.zoo.gps import batch_gp
from stgp import settings


import stgp
from stgp.models import GP

import matplotlib.pyplot as plt

# Construct Data
XS, X, Y = single_output_timeseries(100, 1000, seed=0)

Y[20:30] = np.NaN

print(f"X: {X.shape}, Y: {Y.shape}")


def gp():
    kern = ScaleKernel(Matern52(input_dim=1, lengthscales=[0.1]))
    m = batch_gp(
        X=X,
        Y=Y,
        kernel=kern,
        likelihood=Gaussian(variance=0.1),
    )
    return m


def cvi_gp(parameterisation):
    """
    CVI-GP with a standard GP surrogate model
    Args:
        parameterisation:
            [covariance] - the surrogate GP likelihood will be parameterised using covariances
            [precision] - the surrogate GP likelihood will be parameterised using precisions
    """
    # Construct Model
    Q = 1
    sparsity = stgp.sparsity.NoSparsity(Z=X)

    kern = ScaleKernel(Matern52(input_dim=1, lengthscales=[0.1]))
    latent_gps = [GP(sparsity=sparsity, kernel=kern, prior=True)]

    if parameterisation == "covariance":
        cvi_class = ConjugateGaussian
    elif parameterisation == "precision":
        cvi_class = ConjugatePrecisionGaussian
    else:
        raise NotImplementedError()

    data = Data(X, Y)
    m = GP(
        data=data,
        prior=Independent(latent_gps),
        likelihood=GaussianProductLikelihood([Gaussian(variance=0.1)]),
        approximate_posterior=MeanFieldConjugateGaussian(
            [
                cvi_class(
                    X=sparsity,
                    num_blocks=data.N,
                    block_size=1,
                    num_latents=1,
                    surrogate_model=lambda X, Y, likelihood: stgp.models.GP(
                        data=Data(
                            X.X, Y
                        ),  # Data should already be in the correct format
                        prior=Independent([latent_gps[q]]),
                        likelihood=[likelihood],
                    ),
                )
                for q in range(Q)
            ]
        ),
        inference="Variational",
    )
    return m


def cvi_sde_gp(parallel=False, parameterisation="covariance", gaussian_newton_st=None):
    """CVI-GP with a state-space GP surrogate model parameterised using moment parameterisation."""
    # Construct Model

    if gaussian_newton_st is not None:
        settings.cvi_ng_exploit_space_time = gaussian_newton_st
    else:
        settings.cvi_ng_exploit_space_time = True  # default setting

    Q = 1
    sparsity = stgp.sparsity.NoSparsity(Z=X)

    kern = ScaledMatern52(input_dim=1, lengthscales=[0.1], variance=1.0)
    latent_gps = [GP(sparsity=sparsity, kernel=kern, prior=True)]

    if parallel:
        filter_type = "parallel"
    else:
        filter_type = "sequential"

    if parameterisation == "covariance":
        cvi_class = ConjugateGaussian
    elif parameterisation == "precision":
        cvi_class = ConjugatePrecisionGaussian
    else:
        raise NotImplementedError()

    # data = Data(X, Y)
    data = TemporalData(X, Y)
    m = GP(
        data=data,
        prior=Independent(latent_gps),
        likelihood=GaussianProductLikelihood([Gaussian(variance=0.1)]),
        approximate_posterior=MeanFieldConjugateGaussian(
            [
                cvi_class(
                    X=sparsity,
                    num_blocks=data.N,
                    block_size=1,
                    num_latents=1,
                    surrogate_model=lambda X, Y, likelihood: stgp.models.GP(
                        data=TemporalData(
                            X.X, Y, sort=False
                        ),  # Data should already be in the correct format
                        prior=LTI_SDE(Independent([latent_gps[q]])),
                        likelihood=likelihood,
                        inference="Sequential",
                        filter_type=filter_type,
                    ),
                )
                for q in range(Q)
            ]
        ),
        inference="Variational",
    )
    return m


models = {
    #'cvi_gp_precision': cvi_gp(parameterisation='precision'),
    #'cvi_gp': cvi_gp(parameterisation='covariance'),
    #'cvi_sde_gp_seq_precision': cvi_sde_gp(parallel=False, parameterisation='precision'),
    "gp": {
        "model": gp(),
        "ng": False,
    },
    "cvi_sde_gp_seq_covariance": {
        "model": cvi_sde_gp(parallel=False, parameterisation="covariance"),
        "ng": True,
        "ng_enforce_type": None,
        "gaussian_newton_st": None,
    },
    "cvi_sde_gp_seq_covariance_gauss_newton_st": {
        "model": cvi_sde_gp(
            parallel=False, parameterisation="covariance", gaussian_newton_st=True
        ),
        "ng": True,
        "ng_enforce_type": "laplace_gauss_newton_delta_u",
        "gaussian_newton_st": True,
    },
    "cvi_sde_gp_seq_covariance_gauss_newton": {
        "model": cvi_sde_gp(
            parallel=False, parameterisation="covariance", gaussian_newton_st=False
        ),
        "ng": True,
        "ng_enforce_type": "laplace_gauss_newton_delta_u",
        "gaussian_newton_st": False,
    },
    "cvi_sde_gp_parallel_covariance": {
        "model": cvi_sde_gp(parallel=True, parameterisation="covariance"),
        "ng": True,
        "ng_enforce_type": None,
        "gaussian_newton_st": None,
    },
    "cvi_sde_gp_parallel_covariance_gauss_newton": {
        "model": cvi_sde_gp(parallel=True, parameterisation="covariance"),
        "ng": True,
        "ng_enforce_type": "laplace_gauss_newton_delta_u",
        "gaussian_newton_st": False,
    },
    "cvi_sde_gp_parallel_covariance_gauss_newton_st": {
        "model": cvi_sde_gp(parallel=True, parameterisation="covariance"),
        "ng": True,
        "ng_enforce_type": "laplace_gauss_newton_delta_u",
        "gaussian_newton_st": True,
    },
}

if True:
    for k, m in models.items():
        if m["ng"]:
            if m["gaussian_newton_st"] is not None:
                settings.cvi_ng_exploit_space_time = m["gaussian_newton_st"]
            else:
                settings.cvi_ng_exploit_space_time = True  # default setting

            ng_trainer = NatGradTrainer(
                m["model"], enforce_psd_type=m["ng_enforce_type"]
            )
            ng_trainer.train(1.0, 1)

# check that the ELBOs remain the same after
if True:
    for k, m in models.items():
        m = m["model"]
        print(f"{k}: {m.get_objective()}")

N_models = len(models)

fig, axes = plt.subplots(N_models, 1, squeeze=False)

for i, key in enumerate(models):
    m = models[key]["model"]

    ax_i = axes[i][0]

    pred_mu, pred_var = m.predict_f(XS)
    pred_mu = np.squeeze(pred_mu)
    pred_var = np.squeeze(pred_var)

    ax_i.fill_between(
        np.squeeze(XS),
        np.squeeze(pred_mu - 1.96 * np.sqrt(pred_var)),
        np.squeeze(pred_mu + 1.96 * np.sqrt(pred_var)),
        facecolor=colors.LINE_COL,
        alpha=0.3,
    )
    ax_i.plot(XS, pred_mu, color=colors.LINE_COL, label="GP Fit")
    ax_i.scatter(X, Y, color="black", label="Training Data")
    ax_i.set_title(key)
    ax_i.legend()
plt.show()
