
import jax
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', True)


import numpy as np
import sys
from probnum.diffeq import probsolve_ivp
import matplotlib.pyplot as plt
from probnum.problems import InitialValueProblem
from probnum.randprocs.markov.continuous import ConstantDiffusion
from probnum.randprocs.markov.integrator import IntegratedWienerProcess 
from probnum.diffeq import _utils, odefilter
from probnum import randvars
from probnum import filtsmooth, problems, randprocs, randvars

import stgp
from stgp.models import GP
from stgp.kernels import IntegratedWiener, Wiener, WienerVelocity, Matern32
from stgp.likelihood import Gaussian, ReshapedGaussian, DiagonalGaussian
from stgp.trainers.standard import LBFGS
from stgp.trainers.callbacks import progress_bar_callback
from stgp.data import TemporalData, MultiOutputTemporalData
from stgp.transforms import Independent
from stgp.transforms.sdes import LTI_SDE
from stgp.transforms.pdes import  LotkaVolterra

def f(t, y):
    y1, y2 = y
    return np.array([0.5 * y1 - 0.05 * y1 * y2, -0.5 * y2 + 0.05 * y1 * y2])


def df(t, y):
    y1, y2 = y
    return np.array([[0.5 - 0.05 * y2, -0.05 * y1], [0.05 * y2, -0.5 + 0.05 * y1]])


t0 = 0.0
tmax = 40.0
y0 = np.array([20, 20])

algo_order=1
method="EK1"
adaptive=True
step=0.1
diffusion_model='constant'
atol = 1e-2
rtol = 1e-2

ivp = InitialValueProblem(t0=t0, tmax=tmax, y0=np.asarray(y0), f=f, df=df)
steprule = _utils.construct_steprule(
    ivp=ivp, adaptive=adaptive, step=step, atol=atol, rtol=rtol
)

# Construct diffusion model.
diffusion_model = diffusion_model.lower()
if diffusion_model not in ["constant", "dynamic"]:
    raise ValueError("Diffusion model is not supported.")

if diffusion_model == "constant":
    diffusion = ConstantDiffusion()
else:
    diffusion = randprocs.markov.continuous.PiecewiseConstantDiffusion(t0=ivp.t0)

# Create solver
prior_process = IntegratedWienerProcess(
    initarg=ivp.t0,
    num_derivatives=algo_order,
    wiener_process_dimension=ivp.dimension,
    diffuse=True,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)
solver = odefilter.ODEFilter(
    steprule=steprule,
    prior_process=prior_process,
    approx_strategy=odefilter.approx_strategies.EK0(),
    with_smoothing=True,
    diffusion_model=diffusion,
)

num_steps = 10
x_est, y_est = solver.init_routine._data(
    ivp=ivp, 
    t_eval=np.linspace(
        ivp.t0, 
        ivp.t0+num_steps*solver.init_routine._dt, 
        num_steps, 
        endpoint=True
    ) 
)
ts, ys = x_est, y_est

# Measurement model for SciPy observations
ode_dim = prior_process.transition.wiener_process_dimension
proj_to_y = prior_process.transition.proj2coord(coord=0)
observation_noise_std = solver.init_routine._observation_noise_std * np.ones(ode_dim)
process_noise = randvars.Normal(
    mean=np.zeros(ode_dim),
    cov=np.diag(observation_noise_std**2),
    cov_cholesky=np.diag(observation_noise_std),
)
measmod_scipy = randprocs.markov.discrete.LTIGaussian(
    transition_matrix=proj_to_y,
    noise=process_noise,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)
regression_problem = problems.TimeSeriesRegressionProblem(
    observations=ys, locations=ts, measurement_models=measmod_scipy
)



if False:
    # Infer the solution
    kalman = filtsmooth.gaussian.Kalman(prior_process)
    out, _ = kalman.filter(regression_problem)
    estimated_initrv = out.states[0]

    plt.scatter(ts, np.array([out.states[i][0].mean for i in range(ts.shape[0])]))
    plt.scatter(x_est, y_est[:, 0])
    plt.show()

def get_iwp(X, Y, q=1, var=1.0):
    data = MultiOutputTemporalData(X, Y)
    lik = ReshapedGaussian(DiagonalGaussian(variance=[1e-7, 1e-7]), num_blocks=data.Nt, block_size=2)

    latent_gp = Independent([
        GP(
            sparsity = stgp.sparsity.NoSparsity(Z_ref = data.X), 
            kernel =  IntegratedWiener(q=q, variance=var, m_init=[Y[0][i], 0.0]),
            prior = True
        )
        for i in range(2)
    ])

    prior = LTI_SDE(latent_gp) 

    m = GP(
        data = data,
        prior = prior,
        likelihood = lik,
        inference='Sequential'
    )
    return m
X_est = x_est[:, None]
m = get_iwp(X_est, y_est)
m.get_objective()

pred_mu, pred_var = m.predict_f(X_est, filter_only=True, force_full_state=False)


pred_mu = np.squeeze(pred_mu)
pred_var = np.squeeze(pred_var)

plt.plot(x_est, pred_mu[:, 0])
plt.scatter(x_est, y_est[:, 0])
plt.show()

