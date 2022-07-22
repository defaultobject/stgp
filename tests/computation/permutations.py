"""
Unittests for data-latent permutatations.
"""

import pytest
import numpy as np

import stgp
from stgp.computation.permutations import data_order_to_output_order

from ..common_fixtures import regression_1d_data, gaussian_likelihood, rbf_1d_kernel, gaussian_approximate_posterior


@pytest.mark.parametrize('N', [10, 1])
@pytest.mark.parametrize('num_outputs', [10, 1, 24])
def test__data_order_to_output_order(num_outputs, N):
    # ==== Arrange ====

    # for each output create N datapoints with value corresponding to the output
    v_latent_data = np.zeros([num_outputs, N])
    v_latent_data += np.arange(num_outputs)[:, None]
    v_latent_data = np.hstack(v_latent_data)[:, None]

    v_data_latent = np.tile(np.arange(num_outputs), [N, 1])
    v_data_latent = np.hstack(v_data_latent)[:, None]

    # ==== Act ====
    P = data_order_to_output_order(
        num_outputs,
        N
    )

    v_test = P @ v_latent_data


    # ==== Assert ====
    np.testing.assert_allclose(v_data_latent, v_test)


