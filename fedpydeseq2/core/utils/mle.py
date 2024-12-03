import numpy as np
from pydeseq2.utils import dnb_nll
from pydeseq2.utils import nb_nll

from fedpydeseq2.core.utils.negative_binomial import grid_nb_nll
from fedpydeseq2.core.utils.negative_binomial import vec_nb_nll_grad


def vec_loss(
    counts: np.ndarray,
    design: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    cr_reg: bool = True,
    prior_reg: bool = False,
    alpha_hat: np.ndarray | None = None,
    prior_disp_var: float | None = None,
) -> np.ndarray:
    """Compute the adjusted negative log likelihood of a batch of genes.

    Includes Cox-Reid regularization and (optionally) prior regularization.

    Parameters
    ----------
    counts : ndarray
        Raw counts for a set of genes (n_samples x n_genes).

    design : ndarray
        Design matrix (n_samples x n_params).

    mu : ndarray
        Mean estimation for the NB model (n_samples x n_genes).

    alpha : ndarray
        Dispersion estimates (n_genes).

    cr_reg : bool
        Whether to include Cox-Reid regularization (default: True).

    prior_reg : bool
        Whether to include prior regularization (default: False).

    alpha_hat : ndarray, optional
        Reference dispersions (for MAP estimation, n_genes).

    prior_disp_var : float, optional
        Prior dispersion variance.

    Returns
    -------
    ndarray
        Adjusted negative log likelihood (n_genes).
    """
    # closure to be minimized
    reg = 0
    if cr_reg:
        W = mu / (1 + mu * alpha)
        reg += (
            0.5
            * np.linalg.slogdet((design.T[:, :, None] * W).transpose(2, 0, 1) @ design)[
                1
            ]
        )
    if prior_reg:
        if prior_disp_var is None:
            raise ValueError("Sigma_prior is required for prior regularization")
        reg += (np.log(alpha) - np.log(alpha_hat)) ** 2 / (2 * prior_disp_var)
    return nb_nll(counts, mu, alpha) + reg


def local_grid_summands(
    counts: np.ndarray,
    design: np.ndarray,
    mu: np.ndarray,
    alpha_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute local summands of the adjusted negative log likelihood on a grid.

    Includes the Cox-Reid regularization.

    Parameters
    ----------
    counts : ndarray
        Raw counts for a set of genes (n_samples x n_genes).

    design : ndarray
        Design matrix (n_samples x n_params).

    mu : ndarray
        Mean estimation for the NB model (n_samples x n_genes).

    alpha_grid : ndarray
        Dispersion estimates (n_genes x grid_length).

    Returns
    -------
    nll : ndarray
        Negative log likelihoods of size (n_genes x grid_length).

    cr_matrix : ndarray
        Summands for the Cox-Reid adjustment
        (n_genes x grid_length x n_params x n_params).
    """
    # W is of size (n_samples x n_genes x grid_length)
    W = mu[:, :, None] / (1 + mu[:, :, None] * alpha_grid)
    # cr_matrix is of size (n_genes x grid_length x n_params x n_params)
    cr_matrix = (design.T[:, :, None, None] * W).transpose(2, 3, 0, 1) @ design[
        None, None, :, :
    ]
    # cr_matrix is of size (n_genes x grid_length)
    nll = grid_nb_nll(counts, mu, alpha_grid)

    return nll, cr_matrix


def global_grid_cr_loss(
    nll: np.ndarray,
    cr_grid: np.ndarray,
) -> np.ndarray:
    """Compute the global negative log likelihood on a grid.

    Sums previously computed local negative log likelihoods and Cox-Reid adjustments.

    Parameters
    ----------
    nll : ndarray
        Negative log likelihoods of size (n_genes x grid_length).

    cr_grid : ndarray
        Summands for the Cox-Reid adjustment
        (n_genes x grid_length x n_params x n_params).

    Returns
    -------
    ndarray
        Adjusted negative log likelihood (n_genes x grid_length).
    """
    if np.any(np.isnan(cr_grid)):
        n_genes, grid_length, n_params, _ = cr_grid.shape
        cr_grid = cr_grid.reshape(-1, n_params, n_params)
        mask_nan = np.any(np.isnan(cr_grid), axis=(1, 2))
        slogdet = np.zeros(n_genes * grid_length, dtype=cr_grid.dtype)
        slogdet[mask_nan] = np.nan
        if np.any(~mask_nan):
            slogdet[~mask_nan] = np.linalg.slogdet(cr_grid[~mask_nan])[1]
        return nll + 0.5 * slogdet.reshape(n_genes, grid_length)
    else:
        return nll + 0.5 * np.linalg.slogdet(cr_grid)[1]


def single_mle_grad(
    counts: np.ndarray, design: np.ndarray, mu: np.ndarray, alpha: float
) -> tuple[float, np.ndarray, np.ndarray]:
    r"""Estimate the local gradients of a negative binomial GLM wrt dispersions.

    Returns both the gradient of the negative likelihood, and two matrices used to
    compute the gradient of the Cox-Reid adjustment.


    Parameters
    ----------
    counts : ndarray
        Raw counts for a given gene (n_samples).

    design : ndarray
        Design matrix (n_samples x n_params).

    mu : ndarray
        Mean estimation for the NB model (n_samples).

    alpha : float
        Initial dispersion estimate (1).

    Returns
    -------
    grad : ndarray
        Gradient of the negative log likelihood of the observations counts following
        :math:`NB(\\mu, \\alpha)` (1).

    M1 : ndarray
        First summand for the gradient of the CR adjustment (n_params x n_params).

    M2 : ndarray
        Second summand for the gradient of the CR adjustment (n_params x n_params).
    """
    grad = alpha * dnb_nll(counts, mu, alpha)
    W = mu / (1 + mu * alpha)
    dW = -(W**2)
    M1 = (design.T * W) @ design
    M2 = (design.T * dW) @ design

    return grad, M1, M2


def batch_mle_grad(
    counts: np.ndarray, design: np.ndarray, mu: np.ndarray, alpha: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Estimate the local gradients wrt dispersions on a batch of genes.

    Returns both the gradient of the negative likelihood, and two matrices used to
    compute the gradient of the Cox-Reid adjustment.


    Parameters
    ----------
    counts : ndarray
        Raw counts for a set of genes (n_samples x n_genes).

    design : ndarray
        Design matrix (n_samples x n_params).

    mu : ndarray
        Mean estimation for the NB model (n_samples x n_genes).

    alpha : float
        Initial dispersion estimate (nn_genes).

    Returns
    -------
    grad : ndarray
        Gradient of the negative log likelihood of the observations counts following
        :math:`NB(\\mu, \\alpha)` (n_genes).

    M1 : ndarray
        First summand for the gradient of the CR adjustment
        (n_genes x n_params x n_params).

    M2 : ndarray
        Second summand for the gradient of the CR adjustment
        (n_genes x n_params x n_params).
    """
    grad = alpha * vec_nb_nll_grad(
        counts,
        mu,
        alpha,
    )  # Need to multiply by alpha to get the gradient wrt log_alpha

    W = mu / (1 + mu * alpha[None, :])

    dW = -(W**2)
    M1 = (design.T[:, :, None] * W).transpose(2, 0, 1) @ design[None, :, :]
    M2 = (design.T[:, :, None] * dW).transpose(2, 0, 1) @ design[None, :, :]

    return grad, M1, M2


def batch_mle_update(
    log_alpha: np.ndarray,
    global_CR_summand_1: np.ndarray,
    global_CR_summand_2: np.ndarray,
    global_ll_grad: np.ndarray,
    lr: float,
    alpha_hat: np.ndarray | None = None,
    prior_disp_var: float | None = None,
    prior_reg: bool = False,
):
    """Perform a global dispersions update on a batch of genes.

    Parameters
    ----------
    log_alpha : ndarray
        Current global log dispersions (n_genes).

    global_CR_summand_1 : ndarray
        Global summand 1 for the CR adjustment (n_genes x n_params x n_params).

    global_CR_summand_2 : ndarray
        Global summand 2 for the CR adjustment (n_genes x n_params x n_params).

    global_ll_grad : ndarray
        Global gradient of the negative log likelihood (n_genes).

    lr : float
        Learning rate.

    alpha_hat : ndarray
        Reference dispersions (for MAP estimation, n_genes).

    prior_disp_var : float
        Prior dispersion variance.

    prior_reg : bool
        Whether to use prior regularization for MAP estimation (default: ``False``).

    Returns
    -------
    ndarray
        Updated global log dispersions (n_genes).

    """
    # Add prior regularization, if required
    if prior_reg:
        global_ll_grad += (log_alpha - np.log(alpha_hat)) / prior_disp_var

    # Compute CR reg grad (not separable, cannot be computed locally)
    global_CR_grad = np.array(
        0.5
        * (np.linalg.inv(global_CR_summand_1) * global_CR_summand_2).sum(1).sum(1)
        * np.exp(log_alpha)
    )

    # Update dispersion
    global_log_alpha = log_alpha - lr * (global_ll_grad + global_CR_grad)

    return global_log_alpha
