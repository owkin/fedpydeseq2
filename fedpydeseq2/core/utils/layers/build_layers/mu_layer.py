"""Module to construct mu layer from LFC estimates."""

import anndata as ad
import numpy as np
from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend


def can_set_mu_layer(
    local_adata: ad.AnnData,
    lfc_param_name: str,
    mu_param_name: str,
    raise_error: bool = False,
) -> bool:
    """Check if the mu layer can be reconstructed.

    Parameters
    ----------
    local_adata : ad.AnnData
        The local AnnData object.

    lfc_param_name : str
        The name of the log fold changes parameter in the adata.

    mu_param_name : str
        The name of the mu parameter in the adata.

    raise_error : bool, optional
        If True, raise an error if the mu layer cannot be reconstructed.

    Returns
    -------
    bool
        True if the mu layer can be reconstructed, False otherwise.

    Raises
    ------
    ValueError
        If the mu layer cannot be reconstructed and raise_error is True.
    """
    if mu_param_name in local_adata.layers.keys():
        return True

    has_design_matrix = "design_matrix" in local_adata.obsm.keys()
    has_lfc_param = lfc_param_name in local_adata.varm.keys()
    has_size_factors = "size_factors" in local_adata.obsm.keys()
    has_non_zero = "non_zero" in local_adata.varm.keys()

    has_all = has_design_matrix and has_lfc_param and has_size_factors and has_non_zero
    if not has_all:
        if raise_error:
            raise ValueError(
                "Local adata must contain the design matrix obsm"
                f", the {lfc_param_name} varm to compute the mu layer, "
                f"the size_factors obsm and the non_zero varm. "
                " Here are the keys present in the local adata: "
                f"obsm : {local_adata.obsm.keys()} and varm : {local_adata.varm.keys()}"
            )
        return False
    return True


def set_mu_layer(
    local_adata: ad.AnnData,
    lfc_param_name: str,
    mu_param_name: str,
    n_jobs: int = 1,
    joblib_verbosity: int = 0,
    joblib_backend: str = "loky",
    batch_size: int = 100,
):
    """Reconstruct a mu layer from the adata and a given LFC field.

    Parameters
    ----------
    local_adata : ad.AnnData
        The local AnnData object.

    lfc_param_name : str
        The name of the log fold changes parameter in the adata.

    mu_param_name : str
        The name of the mu parameter in the adata.

    n_jobs : int
        Number of jobs to run in parallel.

    joblib_verbosity : int
        Verbosity level of joblib.

    joblib_backend : str
        Joblib backend to use.

    batch_size : int
        Batch size for parallelization.
    """
    can_set_mu_layer(
        local_adata, lfc_param_name, mu_param_name=mu_param_name, raise_error=True
    )
    if mu_param_name in local_adata.layers.keys():
        return
    gene_names = local_adata.var_names[local_adata.varm["non_zero"]]
    beta = local_adata.varm[lfc_param_name].loc[gene_names].to_numpy()
    design_matrix = local_adata.obsm["design_matrix"].values
    size_factors = local_adata.obsm["size_factors"]

    with parallel_backend(joblib_backend):
        res = Parallel(n_jobs=n_jobs, verbose=joblib_verbosity)(
            delayed(make_mu_batch)(
                beta[i : i + batch_size],
                design_matrix,
                size_factors,
            )
            for i in range(0, len(beta), batch_size)
        )

    if len(res) == 0:
        mu = np.zeros((local_adata.shape[0], 0))
    else:
        mu = np.concatenate(list(res), axis=1)

    mu_layer = np.full(local_adata.shape, np.NaN)

    mu_layer[:, local_adata.var_names.get_indexer(gene_names)] = mu

    local_adata.layers[mu_param_name] = mu_layer


def make_mu_batch(
    beta: np.ndarray,
    design_matrix: np.ndarray,
    size_factors: np.ndarray,
) -> np.ndarray:
    """Compute the mu matrix for a batch of LFC estimates.

    Parameters
    ----------
    beta : np.ndarray
        Current LFC estimate, of shape (batch_size, n_params).
    design_matrix : np.ndarray
        The design matrix, of shape (n_obs, n_params).
    size_factors : np.ndarray
        The size factors, of shape (n_obs).

    Returns
    -------
    mu : np.ndarray
        The mu matrix, of shape (n_obs, batch_size).
    """
    mu = size_factors[:, None] * np.exp(design_matrix @ beta.T)

    return mu
