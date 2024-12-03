"""Module to construct the normed_counts layer."""

import anndata as ad


def can_get_normed_counts(adata: ad.AnnData, raise_error: bool = False) -> bool:
    """Check if the normed_counts layer can be reconstructed.

    Parameters
    ----------
    adata : ad.AnnData
        The local AnnData object.

    raise_error : bool, optional
        If True, raise an error if the normed_counts layer cannot be reconstructed.

    Returns
    -------
    bool
        True if the normed_counts layer can be reconstructed, False otherwise.

    Raises
    ------
    ValueError
        If the normed_counts layer cannot be reconstructed and raise_error is True.

    """
    if "normed_counts" in adata.layers.keys():
        return True
    has_X = adata.X is not None
    has_size_factors = "size_factors" in adata.obsm.keys()
    if not has_X or not has_size_factors:
        if raise_error:
            raise ValueError(
                "Local adata must contain the X field "
                "and the size_factors obsm to compute the normed_counts layer."
                " Here are the keys present in the adata: "
                f" obsm : {adata.obsm.keys()}"
            )
        return False
    return True


def set_normed_counts(adata: ad.AnnData):
    """Reconstruct the normed_counts layer.

    Parameters
    ----------
    adata : ad.AnnData
        The local AnnData object.

    """
    can_get_normed_counts(adata, raise_error=True)
    if "normed_counts" in adata.layers.keys():
        return
    adata.layers["normed_counts"] = adata.X / adata.obsm["size_factors"][:, None]
