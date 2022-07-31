""" Unittests for base transformations.  """

import pytest
import numpy as np
import scipy

import stgp
from stgp.transforms import Independent

from ..common_fixtures import regression_1d_data, rbf_1d_kernel, gaussian_likelihood, gp_prior_1d

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

