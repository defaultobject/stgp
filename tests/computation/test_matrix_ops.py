""" Unittests for Matrix Ops """

import pytest
from unittest.mock import MagicMock
import numpy as np
import scipy

from ..common_fixtures import random_covariance_matrix

from stgp.computation.matrix_ops import block_diagonal_from_cholesky, get_block_diagonal



@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('P', [100])
def test__block_diagonal_from_cholesky(seed, P, random_covariance_matrix):
    # ==== Arrange ====
    L, R = random_covariance_matrix

    # ==== Act ====
    test_blocks = block_diagonal_from_cholesky(L, 10)
    true_blocks = get_block_diagonal(R, 10)

    # ==== Assert ====
    np.testing.assert_allclose(true_blocks, test_blocks, rtol=1e-4)
