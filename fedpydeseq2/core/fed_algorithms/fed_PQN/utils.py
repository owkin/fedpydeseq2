"""Utility functions for the proximal Newton optimization.

This optimization is used in the catching of the IRLS algorithm.
"""


import numpy as np
from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend

from fedpydeseq2.core.utils.negative_binomial import mu_grid_nb_nll


def make_fisher_gradient_nll_step_sizes_batch(
    design_matrix: np.ndarray,
    size_factors: np.ndarray,
    beta: np.ndarray,
    dispersions: np.ndarray,
    counts: np.ndarray,
    ascent_direction: np.ndarray | None,
    step_sizes: np.ndarray | None,
    beta_min: float | None,
    beta_max: float | None,
    min_mu: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make local gradient, fisher matrix, and nll for multiple steps.

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
    ascent_direction : np.ndarray
        The ascent direction, of shape (batch_size, n_params).
    step_sizes: np.ndarray
        A list of step sizes to evaluate, of size (n_steps, ).
    beta_min: float
        The minimum value tolerated for beta.
    beta_max: float
        The maximum value tolerated for beta.
    min_mu : float
        Lower bound on estimated means, to ensure numerical stability.

    Returns
    -------
    H : Optional[ndarray]
        The Fisher information matrix, of shape
        (n_steps, batch_size, n_params, n_params).
    gradient : ndarray
        The gradient, of shape (n_steps, batch_size, n_params).
    nll : ndarray
        The nll evaluations on all steps, of size (n_steps, batch_size).
    """
    # If no ascent direction is provided, we do not need to compute the grid
    # of beta values, but only the current beta value, where we unsqueeze the
    # first dimension to make it compatible with the rest of the code
    # This is the case when we are at the first iteration of the optimization
    if ascent_direction is None and step_sizes is None:
        beta_grid = np.clip(
            beta[None, :, :],
            beta_min,
            beta_max,
        )  # of shape (n_steps, batch_size, n_params)

    # In this case, we compute the grid of beta values, by moving in the direction
    # of the ascent direction, by the step sizes
    else:
        assert isinstance(step_sizes, np.ndarray) and isinstance(
            ascent_direction, np.ndarray
        )
        beta_grid = np.clip(
            beta[None, :, :] - step_sizes[:, None, None] * ascent_direction[None, :, :],
            beta_min,
            beta_max,
        )  # of shape (n_steps, batch_size, n_params)

    mu_grid = size_factors[None, None, :] * np.exp(
        (design_matrix[None, None, :, :] @ beta_grid[:, :, :, None]).squeeze(axis=3)
    )  # of shape (n_steps, batch_size, n_obs)
    mu_grid = np.maximum(
        mu_grid,
        min_mu,
    )

    # --- Step 1: Compute the gradient ----#

    gradient_term_1 = -(design_matrix.T @ counts).T[
        None, :, :
    ]  # shape (1, batch_size, n_params)
    gradient_term_2 = (
        design_matrix.T[None, None, :, :]
        @ (
            (1 / dispersions[None, :, None] + counts.T[None, :, :])
            * mu_grid
            / (1 / dispersions[None, :, None] + mu_grid)  # n_steps, batch_size, n_obs
        )[:, :, :, None]
    ).squeeze(
        3
    )  # Shape n_steps, batch_size, n_params
    gradient = gradient_term_1 + gradient_term_2

    # ---- Step 2: Compute the Fisher matrix  ----#

    W = mu_grid / (1.0 + mu_grid * dispersions[None, :, None])
    expanded_design = design_matrix[
        None, None, :, :
    ]  # of shape (1, 1, n_obs, n_params)
    assert W is not None
    H = (expanded_design * W[:, :, :, None]).transpose(0, 1, 3, 2) @ expanded_design
    # H of size (n_steps, batch_size, n_params, n_params)

    # Get the mu_grid
    nll = mu_grid_nb_nll(counts, mu_grid, dispersions)

    return H, gradient, nll


def compute_gradient_scaling_matrix_fisher(
    fisher: np.ndarray,
    backend: str,
    num_jobs: int,
    joblib_verbosity: int,
    batch_size: int,
):
    """Compute the gradient scaling matrix using the Fisher information.

    In this case, we simply invert the provided Fisher matrix to get the gradient
    scaling matrix.

    Parameters
    ----------
    fisher : ndarray
        The Fisher matrix, of shape (n_genes, n_params, n_params)
    backend : str
        The backend to use for parallelization
    num_jobs : int
        The number of cpus to use
    joblib_verbosity : int
        The verbosity level of joblib
    batch_size : int
        The batch size to use for the computation

    Returns
    -------
    ndarray
        The gradient scaling matrix, of shape (n_genes, n_params, n_params)
    """
    with parallel_backend(backend):
        res = Parallel(n_jobs=num_jobs, verbose=joblib_verbosity)(
            delayed(np.linalg.inv)(
                fisher[i : i + batch_size],
            )
            for i in range(0, len(fisher), batch_size)
        )
    if len(res) > 0:
        gradient_scaling_matrix = np.concatenate(res)
    else:
        gradient_scaling_matrix = np.zeros_like(fisher)

    return gradient_scaling_matrix


def compute_ascent_direction_decrement(
    gradient_scaling_matrix: np.ndarray,
    gradient: np.ndarray,
    beta: np.ndarray,
    max_beta: float,
):
    """Compute the ascent direction and decrement.

    We do this from the gradient scaling matrix, the gradient,
    the beta and the max beta, which embodies the box constraints.

    Please look at this paper for the precise references to the equations:
    https://www.cs.utexas.edu/~inderjit/public_papers/pqnj_sisc10.pdf

    By ascent direction, we mean that the direction we compute is positively
    correlated with the gradient. As our aim is to minimize the function,
    we want to move in the opposite direction of the ascent direction, but
    it is simpler to compute the ascent direction to avoid sign errors.

    Parameters
    ----------
    gradient_scaling_matrix : np.ndarray
        The gradient scaling matrix, of shape (n_genes, n_params, n_params).
    gradient : np.ndarray
        The gradient per gene, of shape (n_genes, n_params).
    beta : np.ndarray
        Beta on those genes, of shape (n_genes, n_params).
    max_beta : float
        The max absolute value for beta.

    Returns
    -------
    ascent_direction : np.ndarray
        The new ascent direction, of shape (n_genes, n_params).
    newton_decrement : np.ndarray
        The newton decrement associated to these ascent directions
        of shape (n_genes, )
    """
    # ---- Step 1: compute first index set ---- #
    # See https://www.cs.utexas.edu/~inderjit/public_papers/pqnj_sisc10.pdf
    # equation 2.2

    lower_binding = (beta < -max_beta + 1e-14) & (gradient > 0)
    upper_binding = (beta > max_beta - 1e-14) & (gradient < 0)
    first_index_mask = lower_binding | upper_binding  # of shape (n_genes, n_params)

    # Set to zero the gradient scaling matrix on the first index

    n_params = beta.shape[1]

    gradient_scaling_matrix[
        np.repeat(first_index_mask[:, :, None], repeats=n_params, axis=2)
    ] = 0
    gradient_scaling_matrix[
        np.repeat(first_index_mask[:, None, :], repeats=n_params, axis=1)
    ] = 0

    ascent_direction = (gradient_scaling_matrix @ gradient[:, :, None]).squeeze(
        axis=2
    )  # of shape (n_genes, n_params)

    # ---- Step 2: Compute the second index set ---- #
    # See https://www.cs.utexas.edu/~inderjit/public_papers/pqnj_sisc10.pdf
    # equation 2.3

    lower_binding = (beta < -max_beta + 1e-14) & (ascent_direction > 0)
    upper_binding = (beta > max_beta - 1e-14) & (ascent_direction < 0)
    second_index_mask = lower_binding | upper_binding

    # Set to zero the gradient scaling matrix on the second index

    gradient_scaling_matrix[
        np.repeat(second_index_mask[:, :, None], repeats=n_params, axis=2)
    ] = 0
    gradient_scaling_matrix[
        np.repeat(second_index_mask[:, None, :], repeats=n_params, axis=1)
    ] = 0

    # ---- Step 3: Compute the ascent direction and Newton decrement ---- #

    ascent_direction = gradient_scaling_matrix @ gradient[:, :, None]
    newton_decrement = (gradient[:, None, :] @ ascent_direction).squeeze(axis=(1, 2))

    ascent_direction = ascent_direction.squeeze(axis=2)

    return ascent_direction, newton_decrement
