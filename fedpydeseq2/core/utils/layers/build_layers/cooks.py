"""Module to set the cooks layer."""


import anndata as ad
import numpy as np

from fedpydeseq2.core.utils.layers.build_layers.hat_diagonals import (
    can_set_hat_diagonals_layer,
)
from fedpydeseq2.core.utils.layers.build_layers.hat_diagonals import (
    set_hat_diagonals_layer,
)
from fedpydeseq2.core.utils.layers.build_layers.mu_layer import can_set_mu_layer
from fedpydeseq2.core.utils.layers.build_layers.mu_layer import set_mu_layer


def can_set_cooks_layer(
    adata: ad.AnnData, shared_state: dict | None, raise_error: bool = False
) -> bool:
    """Check if the Cook's distance can be set.

    Parameters
    ----------
    adata : ad.AnnData
        The local adata.

    shared_state : Optional[dict]
        The shared state containing the Cook's dispersion values.

    raise_error : bool
        Whether to raise an error if the Cook's distance cannot be set.

    Returns
    -------
    bool:
        Whether the Cook's distance can be set.

    Raises
    ------
    ValueError:
        If the Cook's distance cannot be set and raise_error is True.
    """
    if "cooks" in adata.layers.keys():
        return True
    if shared_state is None:
        if raise_error:
            raise ValueError(
                "To set cooks layer, there should be " "an input shared state"
            )
        else:
            return False
    has_non_zero = "non_zero" in adata.varm.keys()
    try:
        has_hat_diagonals = can_set_hat_diagonals_layer(
            adata, shared_state, raise_error
        )
    except ValueError as hat_diagonals_error:
        raise ValueError(
            "The Cook's distance cannot be set because the hat diagonals cannot be set."
        ) from hat_diagonals_error
    try:
        has_mu_LFC = can_set_mu_layer(
            local_adata=adata,
            lfc_param_name="LFC",
            mu_param_name="_mu_LFC",
        )
    except ValueError as mu_LFC_error:
        raise ValueError(
            "The Cook's distance cannot be set because the mu_LFC layer cannot be set."
        ) from mu_LFC_error
    has_X = adata.X is not None
    has_cooks_dispersions = "cooks_dispersions" in shared_state.keys()
    has_all = (
        has_non_zero
        and has_hat_diagonals
        and has_mu_LFC
        and has_X
        and has_cooks_dispersions
    )
    if not has_all and raise_error:
        raise ValueError(
            "The Cook's distance cannot be set because "
            "the following conditions are not met:"
            f"\n- has_non_zero: {has_non_zero}"
            f"\n- has_hat_diagonals: {has_hat_diagonals}"
            f"\n- has_mu_LFC: {has_mu_LFC}"
            f"\n- has_X: {has_X}"
            f"\n- has_cooks_dispersions: {has_cooks_dispersions}"
        )
    return has_all


def set_cooks_layer(
    adata: ad.AnnData,
    shared_state: dict | None,
):
    """Compute the Cook's distance from the shared state.

    This function computes the Cook's distance from the shared state and stores it
    in the "cooks" layer of the local adata.

    Parameters
    ----------
    adata : ad.AnnData
        The local adata.

    shared_state : dict
        The shared state containing the Cook's dispersion values.
    """
    can_set_cooks_layer(adata, shared_state, raise_error=True)
    if "cooks" in adata.layers.keys():
        return
    # set all necessary layers
    assert isinstance(shared_state, dict)
    set_mu_layer(adata, lfc_param_name="LFC", mu_param_name="_mu_LFC")
    set_hat_diagonals_layer(adata, shared_state)
    num_vars = adata.uns["n_params"]
    cooks_dispersions = shared_state["cooks_dispersions"]
    V = (
        adata[:, adata.varm["non_zero"]].layers["_mu_LFC"]
        + cooks_dispersions[None, adata.varm["non_zero"]]
        * adata[:, adata.varm["non_zero"]].layers["_mu_LFC"] ** 2
    )
    squared_pearson_res = (
        adata[:, adata.varm["non_zero"]].X
        - adata[:, adata.varm["non_zero"]].layers["_mu_LFC"]
    ) ** 2 / V
    diag_mul = (
        adata[:, adata.varm["non_zero"]].layers["_hat_diagonals"]
        / (1 - adata[:, adata.varm["non_zero"]].layers["_hat_diagonals"]) ** 2
    )
    adata.layers["cooks"] = np.full((adata.n_obs, adata.n_vars), np.NaN)
    adata.layers["cooks"][:, adata.varm["non_zero"]] = (
        squared_pearson_res / num_vars * diag_mul
    )
