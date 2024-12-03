"""Gradients and loss functions for the negative binomial distribution."""

import numpy as np
from scipy.special import gammaln  # type: ignore
from scipy.special import polygamma


def vec_nb_nll_grad(
    counts: np.ndarray, mu: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    r"""Return the gradient of the negative log-likelihood of a negative binomial.

    Vectorized version (wrt genes).

    Parameters
    ----------
    counts : ndarray
        Observations, n_samples x n_genes.

    mu : ndarray
        Mean of the distribution.

    alpha : pd.Series
        Dispersion of the distribution, s.t. the variance is
        :math:`\\mu + \\alpha_grid * \\mu^2`.

    Returns
    -------
    ndarray
        Gradient of the negative log likelihood of the observations counts following
        :math:`NB(\\mu, \\alpha_grid)`.
    """
    alpha_neg1 = 1 / alpha
    ll_part = alpha_neg1**2 * (
        polygamma(0, alpha_neg1[None, :])
        - polygamma(0, counts + alpha_neg1[None, :])
        + np.log(1 + mu * alpha[None, :])
        + (counts - mu) / (mu + alpha_neg1[None, :])
    ).sum(0)

    return -ll_part


def grid_nb_nll(
    counts: np.ndarray,
    mu: np.ndarray,
    alpha_grid: np.ndarray,
    mask_nan: np.ndarray | None = None,
) -> np.ndarray:
    r"""Neg log-likelihood of a negative binomial, batched wrt genes on a grid.

    Parameters
    ----------
    counts : ndarray
        Observations, n_samples x n_genes.

    mu : ndarray
        Mean estimation for the NB model (n_samples x n_genes).

    alpha_grid : ndarray
        Dispersions (n_genes x grid_length).

    mask_nan : ndarray
        Mask for the values of the grid where mu should have taken values >> 1.

    Returns
    -------
    ndarray
        Negative log likelihoods of size (n_genes x grid_length).
    """
    n = len(counts)
    alpha_neg1 = 1 / alpha_grid
    ndim_alpha = alpha_grid.ndim
    extra_dims_counts = tuple(range(2, 2 + ndim_alpha - 1))
    expanded_counts = np.expand_dims(counts, axis=extra_dims_counts)
    # In order to avoid infinities, we replace all big values in the mu with 1 and
    # modify the final quantity with their true value for the inputs were mu should have
    # taken values >> 1
    if mask_nan is not None:
        mu[mask_nan] = 1.0
    expanded_mu = np.expand_dims(mu, axis=extra_dims_counts)
    logbinom = (
        gammaln(expanded_counts + alpha_neg1[None, :])
        - gammaln(expanded_counts + 1)
        - gammaln(alpha_neg1[None, :])
    )

    nll = n * alpha_neg1 * np.log(alpha_grid) + (
        -logbinom
        + (expanded_counts + alpha_neg1) * np.log(alpha_neg1 + expanded_mu)
        - expanded_counts * np.log(expanded_mu)
    ).sum(0)
    if mask_nan is not None:
        nll[mask_nan.sum(0) > 0] = np.nan
    return nll


def mu_grid_nb_nll(
    counts: np.ndarray, mu_grid: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    r"""Compute the neg log-likelihood of a negative binomial.

    This function is *batched* wrt genes on a mu grid.

    Parameters
    ----------
    counts : ndarray
        Observations, (n_obs, batch_size).

    mu_grid : ndarray
        Means of the distribution :math:`\\mu`, (n_mu, batch_size, n_obs).

    alpha : ndarray
        Dispersions of the distribution :math:`\\alpha`,
        s.t. the variance is :math:`\\mu + \\alpha \\mu^2`,
        of size (batch_size,).

    Returns
    -------
    ndarray
        Negative log likelihoods of the observations counts
        following :math:`NB(\\mu, \\alpha)`, of size (n_mu, batch_size).

    Notes
    -----
    [1] https://en.wikipedia.org/wiki/Negative_binomial_distribution
    """
    n = len(counts)
    alpha_neg1 = 1 / alpha  # shape (batch_size,)
    logbinom = np.expand_dims(
        (
            gammaln(counts.T + alpha_neg1[:, None])
            - gammaln(counts.T + 1)
            - gammaln(alpha_neg1[:, None])
        ),
        axis=0,
    )  # Of size (1, batch_size, n_obs)
    first_term = np.expand_dims(
        n * alpha_neg1 * np.log(alpha), axis=0
    )  # Of size (1, batch_size)
    second_term = np.expand_dims(
        counts.T + np.expand_dims(alpha_neg1, axis=1), axis=0
    ) * np.log(
        np.expand_dims(alpha_neg1, axis=(0, 2)) + mu_grid
    )  # Of size (n_mu, batch_size, n_obs)
    third_term = -np.expand_dims(counts.T, axis=0) * np.log(
        mu_grid
    )  # Of size (n_mu, batch_size, n_obs)
    return first_term + (-logbinom + second_term + third_term).sum(axis=2)
