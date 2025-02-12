"""Module to set the hat diagonals layer."""

import anndata as ad
import numpy as np
from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend


def can_set_hat_diagonals_layer(
    adata: ad.AnnData, shared_state: dict | None, raise_error: bool = False
) -> bool:
    """Check if the hat diagonals layer can be reconstructed.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object.

    shared_state : Optional[dict]
        The shared state dictionary.

    raise_error : bool, optional
        If True, raise an error if the hat diagonals layer cannot be reconstructed.

    Returns
    -------
    bool
        True if the hat diagonals layer can be reconstructed, False otherwise.

    Raises
    ------
    ValueError
        If the hat diagonals layer cannot be reconstructed and raise_error is True.
    """
    if "_hat_diagonals" in adata.layers.keys():
        return True

    if shared_state is None:
        if raise_error:
            raise ValueError(
                "To set the _hat_diagonals layer, there" "should be a shared state."
            )
        else:
            return False

    has_design_matrix = "design_matrix" in adata.obsm.keys()
    has_lfc_param = "LFC" in adata.varm.keys()
    has_size_factors = "size_factors" in adata.obsm.keys()
    has_non_zero = "non_zero" in adata.varm.keys()
    has_dispersion = "dispersions" in adata.varm.keys()
    has_global_hat_matrix_inv = "global_hat_matrix_inv" in shared_state.keys()

    has_all = (
        has_design_matrix
        and has_lfc_param
        and has_size_factors
        and has_non_zero
        and has_global_hat_matrix_inv
        and has_dispersion
    )
    if not has_all:
        if raise_error:
            raise ValueError(
                "Adata must contain the design matrix obsm"
                ", the LFC varm, the dispersions varm, "
                "the size_factors obsm, the non_zero varm "
                "and the global_hat_matrix_inv "
                "in the shared state to compute the hat diagonals layer."
                " Here are the keys present in the adata: "
                f"obsm : {adata.obsm.keys()} and varm : {adata.varm.keys()}, and the "
                f"shared state keys: {shared_state.keys()}"
            )
        return False
    return True


def set_hat_diagonals_layer(
    adata: ad.AnnData,
    shared_state: dict | None,
    n_jobs: int = 1,
    joblib_verbosity: int = 0,
    joblib_backend: str = "loky",
    batch_size: int = 100,
    min_mu: float = 0.5,
):
    """Compute the hat diagonals layer from the adata and the shared state.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object.

    shared_state : Optional[dict]
        The shared state dictionary.
        This dictionary must contain the global hat matrix inverse.

    n_jobs : int
        The number of jobs to use for parallel processing.

    joblib_verbosity : int
        The verbosity level of joblib.

    joblib_backend : str
        The joblib backend to use.

    batch_size : int
        The batch size for parallel processing.

    min_mu : float
        Lower bound on estimated means, to ensure numerical stability.

    Returns
    -------
    np.ndarray
        The hat diagonals layer, of shape (n_obs, n_params).
    """
    can_set_hat_diagonals_layer(adata, shared_state, raise_error=True)
    if "_hat_diagonals" in adata.layers.keys():
        return

    assert shared_state is not None, (
        "To construct the _hat_diagonals layer, " "one must have a shared state."
    )

    gene_names = adata.var_names[adata.varm["non_zero"]]
    beta = adata.varm["LFC"].loc[gene_names].to_numpy()
    design_matrix = adata.obsm["design_matrix"].values
    size_factors = adata.obsm["size_factors"]

    dispersions = adata[:, gene_names].varm["dispersions"]

    # ---- Step 1: Compute the mu and the diagonal of the hat matrix ---- #

    with parallel_backend(joblib_backend):
        res = Parallel(n_jobs=n_jobs, verbose=joblib_verbosity)(
            delayed(make_hat_diag_batch)(
                beta[i : i + batch_size],
                shared_state["global_hat_matrix_inv"][i : i + batch_size],
                design_matrix,
                size_factors,
                dispersions[i : i + batch_size],
                min_mu,
            )
            for i in range(0, len(beta), batch_size)
        )

    H = np.concatenate(res)

    H_layer = np.full(adata.shape, np.NaN)

    H_layer[:, adata.var_names.get_indexer(gene_names)] = H.T

    adata.layers["_hat_diagonals"] = H_layer


def make_hat_diag_batch(
    beta: np.ndarray,
    global_hat_matrix_inv: np.ndarray,
    design_matrix: np.ndarray,
    size_factors: np.ndarray,
    dispersions: np.ndarray,
    min_mu: float = 0.5,
) -> np.ndarray:
    """Compute the H matrix for a batch of LFC estimates.

    Parameters
    ----------
    beta : np.ndarray
        Current LFC estimate, of shape (batch_size, n_params).
    global_hat_matrix_inv : np.ndarray
        The inverse of the global hat matrix, of shape (batch_size, n_params, n_params).
    design_matrix : np.ndarray
        The design matrix, of shape (n_obs, n_params).
    size_factors : np.ndarray
        The size factors, of shape (n_obs).
    dispersions : np.ndarray
        The dispersions, of shape (batch_size).
    min_mu : float
        Lower bound on estimated means, to ensure numerical stability.
        (default: ``0.5``).

    Returns
    -------
    np.ndarray
        The H matrix, of shape (batch_size, n_obs).
    """
    mu = size_factors[:, None] * np.exp(design_matrix @ beta.T)
    mu_clipped = np.maximum(
        mu,
        min_mu,
    )

    # W of shape (n_obs, batch_size)
    W = mu_clipped / (1.0 + mu_clipped * dispersions[None, :])

    # W_sq Of shape (batch_size, n_obs)
    W_sq = np.sqrt(W).T

    # Inside the diagonal operator is of shape (batch_size, n_obs, n_obs)
    # The diagonal operator takes the diagonal per gene in the batch
    # H is therefore of shape (batch_size, n_obs)
    H = np.diagonal(
        design_matrix @ global_hat_matrix_inv @ design_matrix.T,
        axis1=1,
        axis2=2,
    )

    H = W_sq * H * W_sq

    return H
