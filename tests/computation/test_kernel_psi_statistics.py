import numpy as np

from stgp.computation import kernel_psi_statistics  # noqa: F401
from stgp.computation.kernel_psi_statistics import (
    get_psi_statistics_linear_form_from_mu_var_from_gram,
)
from stgp.dispatch import evoke
from stgp.kernels import RBF, ScaleKernel
from stgp.kernels.bias import ConstantKernel


def test_scale_kernel_psi_statistics_scale_rbf_statistics():
    XS = np.array([[-0.2], [0.7]])
    X = np.array([[-1.0], [0.5], [1.5]])
    noise_m = np.array([[0.1], [0.8]])
    noise_S = np.array([[0.4, 0.1], [0.1, 0.3]])
    variance = 2.5
    rbf = RBF(input_dim=1, lengthscales=np.array([0.9]))
    scaled_rbf = ScaleKernel(rbf, variance)

    rbf_statistics = evoke("kernel_psi_statistics", rbf)(
        XS, X, noise_m, noise_S, rbf
    )
    scaled_statistics = evoke("kernel_psi_statistics", scaled_rbf)(
        XS, X, noise_m, noise_S, scaled_rbf
    )

    for actual, expected in zip(
        scaled_statistics,
        (
            variance * rbf_statistics[0],
            variance**2 * rbf_statistics[1],
            variance * rbf_statistics[2],
        ),
    ):
        np.testing.assert_allclose(actual, expected)


def test_temporal_variance_scales_only_conditional_covariance():
    XS = np.array([[-0.2], [0.7]])
    X = np.array([[-1.0], [0.5], [1.5]])
    Y = np.array([[0.4], [-0.2], [0.7]])
    noise_m = np.array([[0.1], [0.8]])
    noise_S = np.array([[0.4, 0.1], [0.1, 0.3]])
    rbf = RBF(input_dim=1, lengthscales=np.array([0.9]))
    K_xx = rbf.K(X, X) + 1e-6 * np.eye(X.shape[0])
    likelihood_var = np.zeros_like(K_xx)

    def statistics(Y, conditional_covar_scale=1.0):
        return get_psi_statistics_linear_form_from_mu_var_from_gram(
            XS,
            X,
            Y,
            K_ss=None,
            K_sx=None,
            K_xx=K_xx,
            likelihood_var=likelihood_var,
            noise_pred_mu=noise_m,
            noise_pred_var=noise_S,
            base_gp_kernel=rbf,
            conditional_covar_scale=conditional_covar_scale,
        )

    weights, unit_variance = statistics(Y)
    _, conditional_covar = statistics(np.zeros_like(Y))
    uncertain_mean_covar = unit_variance - conditional_covar

    temporal_kernel = ConstantKernel(variance=2.5)
    time = np.array([[0.0]])
    K_tt = np.squeeze(temporal_kernel.K_diag(time))
    scaled_weights, scaled_variance = statistics(Y, K_tt)

    np.testing.assert_allclose(scaled_weights, weights)
    np.testing.assert_allclose(
        scaled_variance,
        K_tt * conditional_covar + uncertain_mean_covar,
    )
