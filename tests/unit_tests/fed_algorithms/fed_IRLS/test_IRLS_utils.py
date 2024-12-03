"""Module to test the base functions of the IRLS algorithm."""

import numpy as np

from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_lfc.utils import (
    make_irls_nll_batch,
)
from fedpydeseq2.core.fed_algorithms.fed_irls.utils import (
    make_irls_update_summands_and_nll_batch,
)


def test_make_irls_update_summands_and_nll_batch():
    """Test the function make_irls_update_summands_and_nll_batch.

    This test checks that the function returns the correct output shapes.
    given input shapes of size (3, 2), (3,), (5, 2), (5,), (3, 5), and a scalar.
    """
    # Create fake data
    design_matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    size_factors = np.array([1.0, 2.0, 3.0])
    beta = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    dispersions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    counts = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
        ]
    )
    min_mu = 0.1

    # Call the function with the fake data
    H, y, nll = make_irls_update_summands_and_nll_batch(
        design_matrix, size_factors, beta, dispersions, counts, min_mu
    )

    # Check that the outputs are correct
    assert H.shape == (5, 2, 2)
    # Check that H is symmetric
    assert np.allclose(H, H.transpose(0, 2, 1))
    assert y.shape == (5, 2)
    assert nll.shape == (5,)


def test_make_irls_update_summands_and_nll_batch_no_warnings():
    """Test the function make_irls_update_summands_and_nll_batch.

    This test checks that the function returns the correct output shapes.
    given input shapes of size (3, 2), (3,), (5, 2), (5,), (3, 5), and a scalar.
    """
    # Create fake data
    design_matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    size_factors = np.array([1.0, 2.0, 3.0])
    beta = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1000.0, 2000.0]])
    dispersions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    counts = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
        ]
    )
    min_mu = 0.1

    # Call the function with the fake data
    import warnings

    warnings.filterwarnings("error")
    H, y, nll = make_irls_update_summands_and_nll_batch(
        design_matrix, size_factors, beta, dispersions, counts, min_mu
    )

    # Check that the outputs are correct
    assert H.shape == (5, 2, 2)
    # Check that H is symmetric
    assert np.allclose(H, H.transpose(0, 2, 1))
    assert y.shape == (5, 2)
    assert nll.shape == (5,)


def test_make_irls_update_summands_and_nll_batch_single_design():
    """Test the function make_irls_update_summands_and_nll_batch.

    This test checks the border case where the design matrix has only one row, and
    there is only one gene.
    """
    # Create fake data
    design_matrix = np.array([[1.0]])
    size_factors = np.array([1.0])
    beta = np.array([[1.0]])
    dispersions = np.array([1.0])
    counts = np.array([[1.0]])
    min_mu = 0.1

    # Call the function with the fake data
    H, y, nll = make_irls_update_summands_and_nll_batch(
        design_matrix, size_factors, beta, dispersions, counts, min_mu
    )

    # Check that the outputs are correct
    assert H.shape == (1, 1, 1)
    assert y.shape == (1, 1)
    assert nll.shape == (1,)


def test_make_irls_nll_batch_specific_sizes():
    """Test the function make_irls_nll_batch.

    This test checks that the function returns the correct output shapes
    given input shapes of size (5, 2), (3, 2), (3,), (5,), and a scalar.
    """
    # Create fake data
    beta = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    design_matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    size_factors = np.array([1.0, 2.0, 3.0])
    dispersions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    counts = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
        ]
    )
    min_mu = 0.1

    # Call the function with the fake data
    nll = make_irls_nll_batch(
        beta, design_matrix, size_factors, dispersions, counts, min_mu
    )

    # Check that the outputs are correct
    assert nll.shape == (5,)


def test_make_irls_nll_batch_single_dim():
    """Test the function make_irls_nll_batch.

    This test checks the border case where the design matrix has only one row, and
    there is only one gene.
    """
    # Create fake data
    beta = np.array([[1.0]])
    design_matrix = np.array([[1.0]])
    size_factors = np.array([1.0])
    dispersions = np.array([1.0])
    counts = np.array([[1.0]])
    min_mu = 0.1

    # Call the function with the fake data
    nll = make_irls_nll_batch(
        beta, design_matrix, size_factors, dispersions, counts, min_mu
    )

    # Check that the outputs are correct
    assert nll.shape == (1,)
