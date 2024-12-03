import numpy as np
import pytest

from fedpydeseq2.core.fed_algorithms.fed_PQN.utils import (
    compute_ascent_direction_decrement,
)
from fedpydeseq2.core.fed_algorithms.fed_PQN.utils import (
    compute_gradient_scaling_matrix_fisher,
)
from fedpydeseq2.core.fed_algorithms.fed_PQN.utils import (
    make_fisher_gradient_nll_step_sizes_batch,
)


def test_make_fisher_gradient_nll_step_sizes_batch():
    """Test the function make_fisher_gradient_nll_step_sizes_batch.

    This function runs on matrices with n_obs = 3, n_params=2, n_steps = 4,
    n_genes = 5.
    """
    # Create fake data
    design_matrix = np.array([[1, 2], [3, 4], [5, 6]])
    size_factors = np.array([1, 2, 3])
    beta = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    ascent_direction = np.array([[1, 3], [5, 7], [9, 11], [13, 15], [17, 19]])
    dispersions = np.array([1, 2, 3, 4, 5])
    counts = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
    step_sizes = np.array([1, 2, 3, 4])

    min_mu = 0.5
    beta_min = -30.0
    beta_max = 30.0

    # Call the function with the fake data
    H, gradient, nll = make_fisher_gradient_nll_step_sizes_batch(
        design_matrix=design_matrix,
        size_factors=size_factors,
        beta=beta,
        dispersions=dispersions,
        counts=counts,
        ascent_direction=ascent_direction,
        step_sizes=step_sizes,
        beta_min=beta_min,
        beta_max=beta_max,
        min_mu=min_mu,
    )

    # Check that the outputs are correct
    assert H.shape == (4, 5, 2, 2)
    # Check that H is symmetric
    assert np.allclose(H, H.transpose(0, 1, 3, 2))

    assert gradient.shape == (4, 5, 2)
    assert nll.shape == (4, 5)


def test_make_fisher_gradient_nll_step_sizes_batch_none():
    """Test the function make_fisher_gradient_nll_step_sizes_batch.

    This function runs on matrices with n_obs = 3, n_params=2, n_steps = 4,
    n_genes = 5.
    """
    # Create fake data
    design_matrix = np.array([[1, 2], [3, 4], [5, 6]])
    size_factors = np.array([1, 2, 3])
    beta = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    ascent_direction = None
    dispersions = np.array([1, 2, 3, 4, 5])
    counts = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
    step_sizes = None

    min_mu = 0.5
    beta_min = -30.0
    beta_max = 30.0

    # Call the function with the fake data
    H, gradient, nll = make_fisher_gradient_nll_step_sizes_batch(
        design_matrix=design_matrix,
        size_factors=size_factors,
        beta=beta,
        dispersions=dispersions,
        counts=counts,
        ascent_direction=ascent_direction,
        step_sizes=step_sizes,
        beta_min=beta_min,
        beta_max=beta_max,
        min_mu=min_mu,
    )

    # Check that the outputs are correct
    assert H.shape == (1, 5, 2, 2)
    # Check that H is symmetric
    assert np.allclose(H, H.transpose(0, 1, 3, 2))

    assert gradient.shape == (1, 5, 2)
    assert nll.shape == (1, 5)


def test_make_fisher_gradient_nll_step_sizes_batch_single():
    """Test the function make_fisher_gradient_nll_step_sizes_batch.

    This test runs on matrices with only one element.

    """
    # Create fake data
    design_matrix = np.array([[1]])
    size_factors = np.array([1])
    beta = np.array([[1]])
    ascent_direction = np.array([[1]])
    step_sizes = np.array([1])
    dispersions = np.array([1])
    counts = np.array([[1]])

    min_mu = 0.5
    beta_min = -30.0
    beta_max = 30.0

    # Call the function with the fake data
    H, gradient, nll = make_fisher_gradient_nll_step_sizes_batch(
        design_matrix=design_matrix,
        size_factors=size_factors,
        beta=beta,
        dispersions=dispersions,
        counts=counts,
        ascent_direction=ascent_direction,
        step_sizes=step_sizes,
        beta_min=beta_min,
        beta_max=beta_max,
        min_mu=min_mu,
    )

    assert H.shape == (1, 1, 1, 1)
    assert gradient.shape == (1, 1, 1)
    assert nll.shape == (1, 1)


@pytest.mark.parametrize("num_jobs", [1, 2])
def test_compute_gradient_scaling_matrix_fisher_fisher(num_jobs: int):
    """Test the function compute_gradient_scaling_matrix_fisher_fisher."""
    # Create the fisher matrix
    fisher = np.array([[[2, 0], [0, 2]], [[1, 0], [0, 1]]])

    # Call the function
    result = compute_gradient_scaling_matrix_fisher(
        fisher=fisher,
        backend="threading",
        num_jobs=1,
        joblib_verbosity=0,
        batch_size=1,
    )

    # Create the expected result
    expected_result = np.array([[[0.5, 0], [0, 0.5]], [[1, 0], [0, 1]]])

    # Check that the result is correct
    assert np.allclose(result, expected_result)


def test_compute_ascent_direction_decrement():
    """Test the function compute_ascent_direction_decrement."""
    # Create the inputs
    gradient_scaling_matrix = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    gradient = np.array([[2, 3], [4, 5]])
    beta = np.array([[1, 2], [3, 4]])
    max_beta = 5

    ascent_direction, newton_decrement = compute_ascent_direction_decrement(
        gradient_scaling_matrix=gradient_scaling_matrix,
        gradient=gradient,
        beta=beta,
        max_beta=max_beta,
    )

    # Create the expected results
    expected_ascent_direction = np.array([[2, 3], [4, 5]])
    expected_newton_decrement = np.array([13, 41])

    # Check that the results are correct
    assert np.allclose(
        ascent_direction, expected_ascent_direction
    ), "The ascent direction does not match the expected ascent direction"
    assert np.allclose(
        newton_decrement, expected_newton_decrement
    ), "The newton decrement does not match the expected newton decrement"
