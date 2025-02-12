"""Module containing the necessary functions to reconstruct the y_hat layer."""

import anndata as ad


def can_get_y_hat(local_adata: ad.AnnData, raise_error: bool = False) -> bool:
    """Check if the y_hat layer can be reconstructed.

    Parameters
    ----------
    local_adata : ad.AnnData
        The local AnnData object.

    raise_error : bool, optional
        If True, raise an error if the y_hat layer cannot be reconstructed.

    Returns
    -------
    bool
        True if the y_hat layer can be reconstructed, False otherwise.

    Raises
    ------
    ValueError
        If the y_hat layer cannot be reconstructed and raise_error is True.
    """
    if "_y_hat" in local_adata.layers.keys():
        return True
    has_design_matrix = "design_matrix" in local_adata.obsm.keys()
    has_beta_rough_dispersions = "_beta_rough_dispersions" in local_adata.varm.keys()
    if not has_design_matrix or not has_beta_rough_dispersions:
        if raise_error:
            raise ValueError(
                "Local adata must contain the design matrix obsm "
                "and the _beta_rough_dispersions varm to compute the y_hat layer."
                " Here are the keys present in the local adata: "
                f"obsm : {local_adata.obsm.keys()} and varm : {local_adata.varm.keys()}"
            )
        return False
    return True


def set_y_hat(local_adata: ad.AnnData):
    """Reconstruct the y_hat layer.

    Parameters
    ----------
    local_adata : ad.AnnData
        The local AnnData object.
    """
    can_get_y_hat(local_adata, raise_error=True)
    if "_y_hat" in local_adata.layers.keys():
        return
    y_hat = (
        local_adata.obsm["design_matrix"].to_numpy()
        @ local_adata.varm["_beta_rough_dispersions"].T
    )
    local_adata.layers["_y_hat"] = y_hat
