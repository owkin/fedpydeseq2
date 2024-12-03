from itertools import product

import numpy as np
import pytest

from fedpydeseq2.core.utils.mle import global_grid_cr_loss


@pytest.mark.parametrize(
    "n_genes, grid_length, n_params, percentage_nan",
    product([50, 100, 500], [5, 20, 50], [2, 3, 5], [0.1, 0.2, 0.9, 1.0]),
)
def test_global_grid_cr_loss_with_nans(n_genes, grid_length, n_params, percentage_nan):
    """
    Test the global_grid_cr_loss function with NaNs in the input arrays.
    """
    np.random.seed(seed=42)
    n_genes, grid_length, n_params = 10, 15, 2
    percentage_nan = 0.1

    nll = np.random.uniform(size=(n_genes, grid_length))
    mask_nan = np.random.uniform(size=(n_genes, grid_length)) < percentage_nan
    nll[mask_nan] = np.nan
    cr_grid = np.random.uniform(size=(n_genes, grid_length, n_params, n_params))
    mask_nan_cr_grid = np.random.uniform(size=(n_genes, grid_length)) < percentage_nan
    cr_grid[mask_nan_cr_grid] = np.nan

    expected = nll + 0.5 * np.linalg.slogdet(cr_grid)[1]
    true_result = global_grid_cr_loss(nll, cr_grid)
    assert np.array_equal(true_result, expected, equal_nan=True)
