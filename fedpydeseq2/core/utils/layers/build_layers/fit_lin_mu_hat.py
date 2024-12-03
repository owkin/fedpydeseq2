"""Module to reconstruct the fit_lin_mu_hat layer."""

import anndata as ad
import numpy as np

from fedpydeseq2.core.utils.layers.build_layers.y_hat import can_get_y_hat
from fedpydeseq2.core.utils.layers.build_layers.y_hat import set_y_hat


def can_get_fit_lin_mu_hat(local_adata: ad.AnnData, raise_error: bool = False) -> bool:
    """Check if the fit_lin_mu_hat layer can be reconstructed.

    Parameters
    ----------
    local_adata : ad.AnnData
        The local AnnData object.

    raise_error : bool, optional
        If True, raise an error if the fit_lin_mu_hat layer cannot be reconstructed.

    Returns
    -------
    bool
        True if the fit_lin_mu_hat layer can be reconstructed, False otherwise.

    Raises
    ------
    ValueError
        If the fit_lin_mu_hat layer cannot be reconstructed and raise_error is True.

    """
    if "_fit_lin_mu_hat" in local_adata.layers.keys():
        return True
    try:
        y_hat_ok = can_get_y_hat(local_adata, raise_error=raise_error)
    except ValueError as y_hat_error:
        raise ValueError(
            f"Error while checking if y_hat can be reconstructed: {y_hat_error}"
        ) from y_hat_error

    has_size_factors = "size_factors" in local_adata.obsm.keys()
    has_non_zero = "non_zero" in local_adata.varm.keys()
    if not has_size_factors or not has_non_zero:
        if raise_error:
            raise ValueError(
                "Local adata must contain the size_factors obsm "
                "and the non_zero varm to compute the fit_lin_mu_hat layer."
                " Here are the keys present in the local adata: "
                f"obsm : {local_adata.obsm.keys()} and varm : {local_adata.varm.keys()}"
            )
        return False
    return y_hat_ok


def set_fit_lin_mu_hat(local_adata: ad.AnnData, min_mu: float = 0.5):
    """
    Calculate the _fit_lin_mu_hat layer using the provided local data.

    Checks are performed to ensure necessary keys are present in the data.

    Parameters
    ----------
    local_adata : ad.AnnData
        The local anndata object containing necessary keys for computation.
    min_mu : float, optional
        The minimum value for mu, defaults to 0.5.

    """
    can_get_fit_lin_mu_hat(local_adata, raise_error=True)
    if "_fit_lin_mu_hat" in local_adata.layers.keys():
        return
    set_y_hat(local_adata)
    mu_hat = local_adata.obsm["size_factors"][:, None] * local_adata.layers["_y_hat"]
    fit_lin_mu_hat = np.maximum(mu_hat, min_mu)

    fit_lin_mu_hat[:, ~local_adata.varm["non_zero"]] = np.nan
    local_adata.layers["_fit_lin_mu_hat"] = fit_lin_mu_hat
