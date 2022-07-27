""" Unittests for DataLatentPermutation class. """

import pytest
from unittest.mock import MagicMock
import numpy as np
import scipy

import stgp
from stgp.computation.permutations import data_order_to_output_order, permute_vec_blocks, permute_vec, unpermute_vec, permute_mat, unpermute_mat

from ..common_fixtures import regression_2d_data, rbf_2d_kernel, gaussian_likelihood, full_posterior_joint_model_no_sparsity, gp_prior_2d, permutation_vectors, permutation_matrices


@pytest.fixture
def mocked_joint_model(permutation_vectors, permutation_matrices, full_posterior_joint_model_no_sparsity):
    """ 
    Constuct a Joint prior but replace all calls to covar and var with mocked matrices that
    we know the permutations of 
    """

    v_data_latent, v_latent_data = permutation_vectors
    blocks_v_latent_data  = np.reshape(v_latent_data, [5, -1])[..., None]

    mat_data_latent, mat_latent_data = permutation_matrices

    q, likelihood, prior, sparsity, data = full_posterior_joint_model_no_sparsity

    prior.parent.mean = MagicMock(return_value=v_latent_data)

    prior.parent.mean_blocks = MagicMock(
        return_value=blocks_v_latent_data
    )
    prior.parent.covar = MagicMock(return_value=mat_latent_data)

    return q, likelihood, prior, sparsity, data


@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('num_outputs', [5]) # must be 5 because joint model is a 2d diff op
@pytest.mark.parametrize('lik_var', [0.2])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
def test__np_mean_blocks(seed, N, num_outputs, permutation_vectors, permutation_matrices, mocked_joint_model):
    # ==== Arrange ====
    v_data_latent, v_latent_data = permutation_vectors
    mat_data_latent, mat_latent_data = permutation_matrices

    q, likelihood, prior, sparsity, data = mocked_joint_model

    # ==== Act ====
    v_true  = np.reshape(v_latent_data, [5, -1])[..., None]
    v_test = prior.np_mean_blocks(data.X)

    # ==== Assert ====
    # DataLatentPermutation should not change the mean
    np.testing.assert_allclose(v_true, v_test)


@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('num_outputs', [5]) # must be 5 because joint model is a 2d diff op
@pytest.mark.parametrize('lik_var', [0.2])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
def test__mean(seed, N, num_outputs, permutation_vectors, permutation_matrices, mocked_joint_model):
    # ==== Arrange ====
    v_data_latent, v_latent_data = permutation_vectors
    mat_data_latent, mat_latent_data = permutation_matrices

    q, likelihood, prior, sparsity, data = mocked_joint_model

    # ==== Act ====
    v_true  = v_data_latent
    v_test = prior.mean(data.X)

    # ==== Assert ====
    np.testing.assert_allclose(v_true, v_test)

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('num_outputs', [5]) # must be 5 because joint model is a 2d diff op
@pytest.mark.parametrize('lik_var', [0.2])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
def test__np_covar(seed, N, num_outputs, permutation_vectors, permutation_matrices, mocked_joint_model):
    # ==== Arrange ====
    v_data_latent, v_latent_data = permutation_vectors
    mat_data_latent, mat_latent_data = permutation_matrices

    q, likelihood, prior, sparsity, data = mocked_joint_model

    # ==== Act ====
    mat_true  = mat_latent_data
    mat_test = prior.np_covar(data.X, data.X)

    # ==== Assert ====
    np.testing.assert_allclose(mat_true, mat_test)

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [10])
@pytest.mark.parametrize('num_outputs', [5]) # must be 5 because joint model is a 2d diff op
@pytest.mark.parametrize('lik_var', [0.2])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
def test__covar(seed, N, num_outputs, permutation_vectors, permutation_matrices, mocked_joint_model):
    # ==== Arrange ====
    v_data_latent, v_latent_data = permutation_vectors
    mat_data_latent, mat_latent_data = permutation_matrices

    q, likelihood, prior, sparsity, data = mocked_joint_model

    # ==== Act ====
    mat_true  = mat_data_latent
    mat_test = prior.covar(data.X, data.X)

    # ==== Assert ====
    np.testing.assert_allclose(mat_true, mat_test)

@pytest.mark.parametrize('seed', [0])
@pytest.mark.parametrize('N', [5])
@pytest.mark.parametrize('num_outputs', [5]) # must be 5 because joint model is a 2d diff op
@pytest.mark.parametrize('lik_var', [0.2])
@pytest.mark.parametrize('rbf_ls', [0.1])
@pytest.mark.parametrize('rbf_var', [2.3])
def test__b_full_var_blocks(seed, N, num_outputs, full_posterior_joint_model_no_sparsity):
    # ==== Arrange ====
    q, likelihood, prior, sparsity, data = full_posterior_joint_model_no_sparsity

    # ==== Act ====
    X = data.X
    X_batched = np.tile(X, [num_outputs, 1, 1])

    mat_true  = prior.covar(X, X) * np.kron(np.eye(data.N), np.ones([5, 5]))

    mat_test = prior.b_full_var_blocks(X_batched, 1, num_outputs)
    mat_test = scipy.linalg.block_diag(*mat_test)

    # ==== Assert ====
    np.testing.assert_allclose(mat_true, mat_test)
