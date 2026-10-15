from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import stgp

from stgp import settings
from stgp.computation.filters.filter_utils import frozen_weights_compute_psi_statistics
from stgp.data import MultiOutputTemporalData
from stgp.kernels import Matern32, RBF, ScaledMatern32, SpatioTemporalSeperableKernel
from stgp.kernels.bias import ConstantKernel
from stgp.likelihood import DiagonalGaussian, ReshapedGaussian
from stgp.models import GP
from stgp.transforms import Independent, UncertainPredictionInput
from stgp.transforms.sdes import LTI_SDE


def test_frozen_weights_uses_fixed_moment_matched_projection():
    inducing_inputs = jnp.array([[-1.0], [0.5], [2.0]])
    kernel = RBF(input_dim=1, lengthscales=[0.8], active_dims=[0])
    prior = SimpleNamespace(
        state_input_indices=[1, None],
        base_prior=SimpleNamespace(
            parent=[
                SimpleNamespace(
                    kernel=SimpleNamespace(k1=ConstantKernel(variance=1.0), k2=kernel)
                ),
                SimpleNamespace(),
            ]
        ),
    )

    # State ordering is [u_2(Z), f_1].
    inducing_mean = jnp.array([[0.4], [-0.2], [0.7]])
    input_mean = 0.3
    mean = jnp.vstack([inducing_mean, jnp.array([[input_mean]])])
    covariance = jnp.diag(jnp.array([0.5, 0.4, 0.6, 0.2]))
    covariance = covariance.at[0, -1].set(0.1)
    covariance = covariance.at[-1, 0].set(0.1)
    H_blocks = [jnp.eye(3), jnp.ones((1, 1))]

    H, extra_variance, observation_mean = frozen_weights_compute_psi_statistics(
        prior,
        {"t": 0.0},
        mean,
        covariance,
        H_blocks,
        [inducing_inputs, jnp.empty((1, 0))],
    )

    lengthscale_squared = 0.8**2
    input_variance = 0.2
    z = np.asarray(inducing_inputs[:, 0])
    psi = (
        1.0 / np.sqrt(1.0 + input_variance / lengthscale_squared)
        * np.exp(-0.5 * (input_mean - z) ** 2 / (lengthscale_squared + input_variance))
    )
    K_zz = np.array(kernel.K(inducing_inputs, inducing_inputs), copy=True)
    K_zz += settings.jitter * np.eye(len(z))
    weights = np.linalg.solve(K_zz, psi).reshape(1, -1)
    expected_mean = weights @ np.asarray(inducing_mean)
    expected_H = np.hstack([weights, np.zeros((1, 1))])

    np.testing.assert_allclose(observation_mean[0], expected_mean[0], rtol=1e-6)
    np.testing.assert_allclose(H[0], expected_H[0], rtol=1e-6)
    np.testing.assert_array_equal(H[1], np.array([0.0, 0.0, 0.0, 1.0]))
    assert extra_variance.shape == (2, 2)

    # Joint-state coupling is carried by existing cross-covariance, not an
    # additional derivative of the frozen moment-matched weights.
    H_high = np.asarray(H[:1])
    P = np.asarray(covariance)
    innovation_variance = H_high @ P @ H_high.T + 0.1
    gain = P @ H_high.T / innovation_variance
    assert np.abs(gain[-1, 0]) > 1e-8


def test_frozen_weights_runs_joint_filter():
    time = np.linspace(0.0, 1.0, 5)[:, None]
    low = np.sin(time)
    high = 2.0 * low
    data = MultiOutputTemporalData(time, np.hstack([high, low]))
    inducing_inputs = np.linspace(-1.0, 1.0, 3)[:, None]

    high_latent = GP(
        sparsity=stgp.sparsity.FITCSpatialSparsity(data.X_time, inducing_inputs),
        kernel=SpatioTemporalSeperableKernel(
            Matern32(input_dim=1, lengthscales=[1.0], active_dims=[0]),
            RBF(input_dim=1, lengthscales=[1.0], active_dims=[1]),
        ),
        prior=True,
    )
    low_latent = GP(
        sparsity=stgp.sparsity.NoSparsity(Z=data.X),
        kernel=ScaledMatern32(input_dim=1, lengthscales=[1.0], variance=1.0),
        prior=True,
    )
    model = GP(
        data=data,
        prior=UncertainPredictionInput(
            LTI_SDE(Independent([high_latent, low_latent])),
            approximation="frozen_weights",
            state_input_indices=[1, None],
        ),
        likelihood=ReshapedGaussian(
            DiagonalGaussian([0.01, 0.01]),
            num_blocks=data.Nt,
            block_size=2,
        ),
        inference="Sequential",
    )

    assert np.isfinite(model.get_objective())
    assert model.filter_meta()["H"].shape[-2:] == (2, 8)
    prediction_mean, prediction_var = model.predict_f(time)
    assert prediction_mean.shape == (5, 2, 1)
    assert prediction_var.shape == (5, 2, 1, 1)
