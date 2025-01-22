"""Module to construct the sqerror layer."""

import anndata as ad
import numpy as np
import pandas as pd

from fedpydeseq2.core.utils.layers.build_layers.normed_counts import (
    can_get_normed_counts,
)
from fedpydeseq2.core.utils.layers.build_layers.normed_counts import set_normed_counts


def can_get_sqerror_layer(adata: ad.AnnData, raise_error: bool = False) -> bool:
    """Check if the squared error layer can be reconstructed.

    Parameters
    ----------
    adata : ad.AnnData
        The local AnnData object.

    raise_error : bool, optional
        If True, raise an error if the squared error layer cannot be reconstructed.

    Returns
    -------
    bool
        True if the squared error layer can be reconstructed, False otherwise.

    Raises
    ------
    ValueError
        If the squared error layer cannot be reconstructed and raise_error is True.
    """
    if "sqerror" in adata.layers.keys():
        return True
    try:
        has_normed_counts = can_get_normed_counts(adata, raise_error=raise_error)
    except ValueError as normed_counts_error:
        raise ValueError(
            f"Error while checking if normed_counts can be"
            f" reconstructed: {normed_counts_error}"
        ) from normed_counts_error

    has_cell_means = "cell_means" in adata.varm.keys()
    has_cell_obs = "cells" in adata.obs.keys()
    if not has_normed_counts or not has_cell_means or not has_cell_obs:
        if raise_error:
            raise ValueError(
                "Local adata must contain the normed_counts layer, the cells obs, "
                "and the cell_means varm to compute the squared error layer."
                " Here are the keys present in the adata: "
                f"obs : {adata.obs.keys()}, varm : {adata.varm.keys()}"
            )
        return False
    return True


def set_sqerror_layer(local_adata: ad.AnnData):
    """Compute the squared error between the normalized counts and the trimmed mean.

    Parameters
    ----------
    local_adata : ad.AnnData
        Local AnnData. It is expected to have the following fields:
        - layers["normed_counts"]: the normalized counts.
        - varm["cell_means"]: the trimmed mean.
        - obs["cells"]: the cells.
    """
    can_get_sqerror_layer(local_adata, raise_error=True)
    if "sqerror" in local_adata.layers.keys():
        return
    cell_means = local_adata.varm["cell_means"]
    set_normed_counts(local_adata)
    if isinstance(cell_means, pd.DataFrame):
        cells = local_adata.obs["cells"]
        # restrict to the cells that are in the cell means columns
        cells = cells[cells.isin(cell_means.columns)]
        qmat = cell_means[cells].T
        qmat.index = cells.index

        # initialize wiht nans
        layer = np.full_like(local_adata.layers["normed_counts"], np.nan)
        indices = local_adata.obs_names.get_indexer(qmat.index)
        layer[indices, :] = (
            local_adata[qmat.index, :].layers["normed_counts"] - qmat
        ) ** 2
    else:
        layer = (local_adata.layers["normed_counts"] - cell_means[None, :]) ** 2
    local_adata.layers["sqerror"] = layer
