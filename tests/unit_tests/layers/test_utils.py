import numpy as np

from fedpydeseq2.core.utils.layers.build_layers.hat_diagonals import make_hat_diag_batch
from fedpydeseq2.core.utils.layers.build_layers.mu_layer import make_mu_batch
from fedpydeseq2.core.utils.layers.cooks_layer import make_hat_matrix_summands_batch


def test_make_hat_matrix_summands_batch():
    """Test the function make_hat_matrix_summands_batch.

    This test checks that the function returns the correct output shape given input
    shapes of size (3, 2), (3,), (5, 2), (5,), and a scalar.
    """
    # Create fake data
    design_matrix = np.array([[1, 2], [3, 4], [5, 6]])
    size_factors = np.array([1, 2, 3])
    beta = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    dispersions = np.array([1, 2, 3, 4, 5])
    min_mu = 0.1

    H = make_hat_matrix_summands_batch(
        design_matrix, size_factors, beta, dispersions, min_mu
    )

    # Check that the outputs are correct

    assert np.allclose(H, H.transpose(0, 2, 1))

    assert H.shape == (5, 2, 2)


def test_make_hat_matrix_summands_batch_single_dim():
    """Test the function make_hat_matrix_summands_batch.

    This test checks the border case where the design matrix has only one row, and
    there is only one gene.
    """
    design_matrix = np.array([[1]])
    size_factors = np.array([1])
    beta = np.array([[1]])
    dispersions = np.array([1])
    min_mu = 0.1

    H = make_hat_matrix_summands_batch(
        design_matrix, size_factors, beta, dispersions, min_mu
    )

    assert H.shape == (1, 1, 1)


def test_make_mu_batch():
    """Test the function make_mu_batch.

    This test checks that the function returns the correct output shapes given input
    shapes of size (5, 2), (3, 2), and (3,).
    """
    beta = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    design_matrix = np.array([[1, 2], [3, 4], [5, 6]])
    size_factors = np.array([1, 2, 3])

    mu = make_mu_batch(
        beta,
        design_matrix,
        size_factors,
    )

    assert mu.shape == (3, 5)


def test_make_mu_batch_single_dim():
    """Test the function make_irls_mu_and_diag_batch.

    This test checks the border case where the design matrix has only one row, and
    there is only one gene.
    """
    beta = np.array([[1]])
    design_matrix = np.array([[1]])
    size_factors = np.array([1])

    mu = make_mu_batch(
        beta,
        design_matrix,
        size_factors,
    )

    assert mu.shape == (1, 1)


def test_make_hat_diag_batch():
    """Test the function make_hat_diag_batch.

    This test checks that the function returns the correct output shapes given input
    shapes of size (3, 2), (3,), (5, 2), (5,), and a scalar.
    """
    beta = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    global_hat_matrix_inv = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
            [[13, 14], [15, 16]],
            [[17, 18], [19, 20]],
        ]
    )
    design_matrix = np.array([[1, 2], [3, 4], [5, 6]])
    size_factors = np.array([1, 2, 3])
    dispersions = np.array([1, 2, 3, 4, 5])
    min_mu = 0.1

    H = make_hat_diag_batch(
        beta, global_hat_matrix_inv, design_matrix, size_factors, dispersions, min_mu
    )

    assert H.shape == (5, 3)


def test_make_hat_diag_batch_single_dim():
    """Test the function make_hat_diag_batch.

    This test checks the border case where the design matrix has only one row, and
    there is only one gene.
    """
    beta = np.array([[1]])
    global_hat_matrix_inv = np.array([[[1]]])
    design_matrix = np.array([[1]])
    size_factors = np.array([1])
    dispersions = np.array([1])
    min_mu = 0.1

    H = make_hat_diag_batch(
        beta, global_hat_matrix_inv, design_matrix, size_factors, dispersions, min_mu
    )

    assert H.shape == (1, 1)
