"""Module to build the mu_hat layer."""

import anndata as ad

from fedpydeseq2.core.utils.layers.build_layers.fit_lin_mu_hat import (
    can_get_fit_lin_mu_hat,
)
from fedpydeseq2.core.utils.layers.build_layers.fit_lin_mu_hat import set_fit_lin_mu_hat
from fedpydeseq2.core.utils.layers.build_layers.mu_layer import can_set_mu_layer
from fedpydeseq2.core.utils.layers.build_layers.mu_layer import set_mu_layer


def can_get_mu_hat(local_adata: ad.AnnData, raise_error: bool = False) -> bool:
    """Check if the mu_hat layer can be reconstructed.

    Parameters
    ----------
    local_adata : ad.AnnData
        The local AnnData object.

    raise_error : bool, optional
        If True, raise an error if the mu_hat layer cannot be reconstructed.

    Returns
    -------
    bool
        True if the mu_hat layer can be reconstructed, False otherwise.

    Raises
    ------
    ValueError
        If the mu_hat layer cannot be reconstructed and raise_error is True.
    """
    if "_mu_hat" in local_adata.layers.keys():
        return True
    has_num_replicates = "num_replicates" in local_adata.uns
    has_n_params = "n_params" in local_adata.uns
    if not has_num_replicates or not has_n_params:
        if raise_error:
            raise ValueError(
                "Local adata must contain num_replicates in uns field "
                "and n_params in uns field to compute mu_hat."
                " Here are the keys present in the local adata: "
                f"uns : {local_adata.uns.keys()}"
            )
        return False
    # If the number of replicates is not equal to the number of parameters,
    # we need to reconstruct mu_hat from the adata.
    if len(local_adata.uns["num_replicates"]) != local_adata.uns["n_params"]:
        try:
            mu_hat_LFC_ok = can_set_mu_layer(
                local_adata=local_adata,
                lfc_param_name="_mu_hat_LFC",
                mu_param_name="_irls_mu_hat",
                raise_error=raise_error,
            )
        except ValueError as mu_hat_LFC_error:
            raise ValueError(
                "Error while checking if mu_hat_LFC can "
                f"be reconstructed: {mu_hat_LFC_error}"
            ) from mu_hat_LFC_error
        return mu_hat_LFC_ok
    else:
        try:
            fit_lin_mu_hat_ok = can_get_fit_lin_mu_hat(
                local_adata=local_adata,
                raise_error=raise_error,
            )
        except ValueError as fit_lin_mu_hat_error:
            raise ValueError(
                "Error while checking if fit_lin_mu_hat can be "
                f"reconstructed: {fit_lin_mu_hat_error}"
            ) from fit_lin_mu_hat_error
        return fit_lin_mu_hat_ok


def set_mu_hat_layer(local_adata: ad.AnnData):
    """Reconstruct the mu_hat layer.

    Parameters
    ----------
    local_adata: ad.AnnData
        The local AnnData object.
    """
    can_get_mu_hat(local_adata, raise_error=True)
    if "_mu_hat" in local_adata.layers.keys():
        return

    if len(local_adata.uns["num_replicates"]) != local_adata.uns["n_params"]:
        set_mu_layer(
            local_adata=local_adata,
            lfc_param_name="_mu_hat_LFC",
            mu_param_name="_irls_mu_hat",
        )
        local_adata.layers["_mu_hat"] = local_adata.layers["_irls_mu_hat"].copy()
        return
    set_fit_lin_mu_hat(
        local_adata=local_adata,
    )
    local_adata.layers["_mu_hat"] = local_adata.layers["_fit_lin_mu_hat"].copy()
