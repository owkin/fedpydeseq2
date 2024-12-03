"""Module to implement the utilities of the IRLS algorithm.

Most of these functions have the _batch suffix, which means that they are
vectorized to work over batches of genes in the parralel_backend file in
the same module.
"""


import numpy as np

from fedpydeseq2.core.utils.negative_binomial import grid_nb_nll


def make_irls_nll_batch(
    beta: np.ndarray,
    design_matrix: np.ndarray,
    size_factors: np.ndarray,
    dispersions: np.ndarray,
    counts: np.ndarray,
    min_mu: float = 0.5,
) -> np.ndarray:
    """
    Compute the negative binomial log likelihood from LFC estimates.

    Used in ComputeLFC to compute the deviance score. This function is vectorized to
    work over batches of genes.

    Parameters
    ----------
    beta : np.ndarray
        Current LFC estimate, of shape (batch_size, n_params).
    design_matrix : np.ndarray
        The design matrix, of shape (n_obs, n_params).
    size_factors : np.ndarray
        The size factors, of shape (n_obs).
    dispersions : np.ndarray
        The dispersions, of shape (batch_size).
    counts : np.ndarray
        The counts, of shape (n_obs,batch_size).
    min_mu : float
        Lower bound on estimated means, to ensure numerical stability.
        (default: ``0.5``).

    Returns
    -------
    np.ndarray
        Local negative binomial log-likelihoods, of shape
        (batch_size).
    """
    mu = np.maximum(
        size_factors[:, None] * np.exp(design_matrix @ beta.T),
        min_mu,
    )
    return grid_nb_nll(
        counts,
        mu,
        dispersions,
    )
