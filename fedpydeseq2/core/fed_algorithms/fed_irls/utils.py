"""Module to implement the utilities of the IRLS algorithm.

Most of these functions have the _batch suffix, which means that they are vectorized to
work over batches of genes in the parralel_backend file in the same module.
"""

import numpy as np

from fedpydeseq2.core.utils.negative_binomial import grid_nb_nll


def make_irls_update_summands_and_nll_batch(
    design_matrix: np.ndarray,
    size_factors: np.ndarray,
    beta: np.ndarray,
    dispersions: np.ndarray,
    counts: np.ndarray,
    min_mu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make the summands for the IRLS algorithm for a given set of genes.

    Parameters
    ----------
    design_matrix : ndarray
        The design matrix, of shape (n_obs, n_params).
    size_factors : ndarray
        The size factors, of shape (n_obs).
    beta : ndarray
        The log fold change matrix, of shape (batch_size, n_params).
    dispersions : ndarray
        The dispersions, of shape (batch_size).
    counts : ndarray
        The counts, of shape (n_obs,batch_size).
    min_mu : float
        Lower bound on estimated means, to ensure numerical stability.

    Returns
    -------
    H : ndarray
        The H matrix, of shape (batch_size, n_params, n_params).
    y : ndarray
        The y vector, of shape (batch_size, n_params).
    nll : ndarray
        The negative binomial negative log-likelihood, of shape (batch_size).
    """
    max_limit = np.log(1e100)
    design_matrix_time_beta_T = design_matrix @ beta.T
    mask_nan = design_matrix_time_beta_T > max_limit

    # In order to avoid overflow and np.inf, we replace all big values in the
    # design_matrix_time_beta_T with 0., then we carry the computation normally, and
    # we modify the final quantity with their true value for the inputs were
    # exp_design_matrix_time_beta_T should have taken values >> 1
    exp_design_matrix_time_beta_T = np.zeros(
        design_matrix_time_beta_T.shape, dtype=design_matrix_time_beta_T.dtype
    )
    exp_design_matrix_time_beta_T[~mask_nan] = np.exp(
        design_matrix_time_beta_T[~mask_nan]
    )
    mu = size_factors[:, None] * exp_design_matrix_time_beta_T

    mu = np.maximum(mu, min_mu)

    W = mu / (1.0 + mu * dispersions[None, :])

    dispersions_broadcast = np.broadcast_to(
        dispersions, (mu.shape[0], dispersions.shape[0])
    )
    W[mask_nan] = 1.0 / dispersions_broadcast[mask_nan]

    z = np.log(mu / size_factors[:, None]) + (counts - mu) / mu
    z[mask_nan] = design_matrix_time_beta_T[mask_nan] - 1.0

    H = (design_matrix.T[:, :, None] * W).transpose(2, 0, 1) @ design_matrix[None, :, :]
    y = (design_matrix.T @ (W * z)).T

    mu[mask_nan] = np.inf
    nll = grid_nb_nll(counts, mu, dispersions, mask_nan)

    return H, y, nll
