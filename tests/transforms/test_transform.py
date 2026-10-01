""" Unittests for base transformations.  """

import pytest
import numpy as np
import scipy

import stgp
from stgp.transforms import Independent, UncertainPredictionInput
from stgp.computation.filters.filter_utils import _setup_uncertain_inputs_state_and_args

from ..common_fixtures import regression_1d_data, rbf_1d_kernel, gaussian_likelihood, gp_prior_1d


def test__uncertain_prediction_input_can_cache_training_predictions():
    class CountingGP:
        def __init__(self):
            self.posterior_calls = 0

        def posterior_in_latent_data(self):
            self.posterior_calls += 1
            return np.arange(3)[None, :], np.ones((1, 3))

    class Data:
        N = 3

    low_fidelity_gp = CountingGP()
    prior = UncertainPredictionInput(
        object(),
        [low_fidelity_gp],
        cache_training_predictions=True,
    )

    _, args = _setup_uncertain_inputs_state_and_args(
        Data(), prior, None, None, None, {}, {}, is_prediction=False
    )
    _setup_uncertain_inputs_state_and_args(
        Data(), prior, None, None, None, {}, {}, is_prediction=False
    )

    assert low_fidelity_gp.posterior_calls == 1
    np.testing.assert_array_equal(
        args['low_fidelity_prediction_mean'], np.arange(3)[:, None]
    )


def test__uncertain_prediction_input_does_not_cache_by_default():
    class CountingGP:
        posterior_calls = 0

        def posterior_in_latent_data(self):
            self.posterior_calls += 1
            return np.arange(3)[None, :], np.ones((1, 3))

    class Data:
        N = 3

    low_fidelity_gp = CountingGP()
    prior = UncertainPredictionInput(object(), [low_fidelity_gp])

    for _ in range(2):
        _setup_uncertain_inputs_state_and_args(
            Data(), prior, None, None, None, {}, {}, is_prediction=False
        )

    assert low_fidelity_gp.posterior_calls == 2

@pytest.mark.parametrize('N', [10, 1])
@pytest.mark.parametrize('NS', [100])
@pytest.mark.parametrize('num_outputs', [10])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
def test__independent_shapes(N, NS, num_outputs, gp_prior_1d):
    # ==== Arrange ====
    X = np.linspace(0, 1, NS)[:, None]
    all_latents = [gp_prior_1d for q in range(num_outputs)]
    prior = Independent(all_latents)

    # ==== Act & Assert ====
    np.testing.assert_equal(
        prior.mean(X).shape, 
        [NS * num_outputs, 1]
    )

    np.testing.assert_equal(
        prior.mean_blocks(X).shape, 
        [num_outputs, NS , 1]
    )
    np.testing.assert_equal(
        prior.covar_blocks(X, X).shape, 
        [num_outputs, NS, NS]
    )

    np.testing.assert_equal(
        prior.var_blocks(X).shape, 
        [num_outputs , NS, 1]
    )

    np.testing.assert_equal(
        prior.full_var_blocks(X).shape, 
        [num_outputs , NS, NS]
    )
