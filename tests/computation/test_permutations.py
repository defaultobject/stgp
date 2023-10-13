""" Unittests for data-latent permutatations.  """

import pytest
import numpy as np
import scipy

import stgp
from stgp.computation.permutations import data_order_to_output_order, permute_vec_blocks, permute_vec, unpermute_vec, permute_mat, unpermute_mat, ld_to_dl, dl_to_ld, permute_mat_ld_to_dl, permute_mat_dl_to_ld

from ..common_fixtures import permutation_vectors, permutation_matrices

@pytest.mark.parametrize('N', [10, 1])
@pytest.mark.parametrize('num_outputs', [10, 1, 24])
def test__data_order_to_output_order(N, num_outputs, permutation_vectors):
    # ==== Arrange ====
    v_data_latent, v_latent_data = permutation_vectors

    # ==== Act ====
    P = data_order_to_output_order(
        num_outputs,
        N
    )

    v_test = P @ v_latent_data


    # ==== Assert ====
    np.testing.assert_allclose(v_data_latent, v_test)

@pytest.mark.parametrize('N', [10, 1])
@pytest.mark.parametrize('num_outputs', [10, 1, 24])
def test__permute_vec_blocks(N, num_outputs, permutation_vectors):
    # ==== Arrange ====
    v_data_latent, v_latent_data = permutation_vectors
    v_latent_data_blocks = v_latent_data.reshape([num_outputs, N, 1])

    # ==== Act ====
    v_test = permute_vec_blocks(v_latent_data_blocks)

    # ==== Assert ====
    np.testing.assert_allclose(v_data_latent, v_test)

@pytest.mark.parametrize('N', [10, 1])
@pytest.mark.parametrize('num_outputs', [10, 1, 24])
def test__permute_vec(N, num_outputs, permutation_vectors):
    # ==== Arrange ====
    v_data_latent, v_latent_data = permutation_vectors

    # ==== Act ====
    v_test = permute_vec(v_latent_data, num_outputs)

    # ==== Assert ====
    np.testing.assert_allclose(v_data_latent, v_test)

@pytest.mark.parametrize('N', [10, 1])
@pytest.mark.parametrize('num_outputs', [10, 1, 24])
def test__unpermute_vec(N, num_outputs, permutation_vectors):
    # ==== Arrange ====
    v_data_latent, v_latent_data = permutation_vectors

    # ==== Act ====
    v_test = unpermute_vec(v_data_latent, num_outputs)

    # ==== Assert ====
    np.testing.assert_allclose(v_latent_data, v_test)

@pytest.mark.parametrize('N', [10, 1])
@pytest.mark.parametrize('num_outputs', [10, 1, 24])
def test__permute_mat(N, num_outputs, permutation_matrices):
    # ==== Arrange ====
    mat_data_latent, mat_latent_data = permutation_matrices

    # ==== Act ====
    mat_test = permute_mat(mat_latent_data, num_outputs)

    # ==== Assert ====
    np.testing.assert_allclose(mat_data_latent, mat_test)


@pytest.mark.parametrize('N', [10, 1])
@pytest.mark.parametrize('num_outputs', [10, 1, 24])
def test__unpermute_mat(N, num_outputs, permutation_matrices):
    # ==== Arrange ====
    mat_data_latent, mat_latent_data = permutation_matrices

    # ==== Act ====
    mat_test = unpermute_mat(mat_data_latent, num_outputs)

    # ==== Assert ====
    np.testing.assert_allclose(mat_latent_data, mat_test)


@pytest.mark.parametrize('N', [10, 1])
@pytest.mark.parametrize('num_outputs', [10, 1, 24])
def test__ld_to_dl_mat(N, num_outputs, permutation_matrices):
    # ==== Arrange ====
    mat_data_latent, mat_latent_data = permutation_matrices

    # ==== Act ====
    P = ld_to_dl(num_outputs, N)
    mat_test = P @ mat_latent_data @ P.T

    # ==== Assert ====
    np.testing.assert_allclose(mat_data_latent, mat_test)

@pytest.mark.parametrize('N', [10, 1])
@pytest.mark.parametrize('num_outputs', [10, 1, 24])
def test__dl_to_ld_mat(N, num_outputs, permutation_matrices):
    # ==== Arrange ====
    mat_data_latent, mat_latent_data = permutation_matrices

    # ==== Act ====
    P = dl_to_ld(num_outputs, N)
    mat_test = P @ mat_data_latent @ P.T

    # ==== Assert ====
    np.testing.assert_allclose(mat_latent_data, mat_test)

@pytest.mark.parametrize('N', [10, 1])
@pytest.mark.parametrize('num_outputs', [10, 1, 24])
def test__permute_mat_ld_to_dl(N, num_outputs, permutation_matrices):
    # ==== Arrange ====
    mat_data_latent, mat_latent_data = permutation_matrices

    # ==== Act ====
    mat_test= permute_mat_ld_to_dl(mat_latent_data, num_outputs, N)

    # ==== Assert ====
    np.testing.assert_allclose(mat_data_latent, mat_test)

@pytest.mark.parametrize('N', [10, 1])
@pytest.mark.parametrize('num_outputs', [10, 1, 24])
def test__dl_to_ld_mat(N, num_outputs, permutation_matrices):
    # ==== Arrange ====
    mat_data_latent, mat_latent_data = permutation_matrices

    # ==== Act ====
    mat_test = permute_mat_dl_to_ld(mat_data_latent, num_outputs, N)

    # ==== Assert ====
    np.testing.assert_allclose(mat_latent_data, mat_test)



