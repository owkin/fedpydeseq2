from typing import Any

import numpy as np
import pandas as pd


def set_basic_refit_adata(self: Any):
    """Set the basic refit adata from the local adata.

    This function checks that the local adata is loaded and the replaced
    genes are computed and stored in the varm field. It then sets the refit
    adata from the local adata.

    Parameters
    ----------
    self : Any
        The object containing the local adata and the refit adata.

    """
    assert (
        self.local_adata is not None
    ), "Local adata must be loaded before setting the refit adata."
    assert (
        "replaced" in self.local_adata.varm.keys()
    ), "Replaced genes must be computed before setting the refit adata."

    genes_to_replace = pd.Series(
        self.local_adata.varm["replaced"], index=self.local_adata.var_names
    )
    if self.refit_adata is None:
        self.refit_adata = self.local_adata[:, genes_to_replace].copy()
        # Clear the varm field of the refit adata
        self.refit_adata.varm = None
    elif "refitted" not in self.local_adata.varm.keys():
        self.refit_adata.X = self.local_adata[:, genes_to_replace].X.copy()
        self.refit_adata.obsm = self.local_adata.obsm
    else:
        genes_to_refit = pd.Series(
            self.local_adata.varm["refitted"], index=self.local_adata.var_names
        )
        self.refit_adata.X = self.local_adata[:, genes_to_refit].X.copy()
        self.refit_adata.obsm = self.local_adata.obsm


def set_imputed_counts_refit_adata(self: Any):
    """Set the imputed counts in the refit adata.

    This function checks that the refit adata, the local adata, the replaced
    genes, the trimmed mean normed counts, the size factors, the cooks G cutoff,
    and the replaceable genes are computed and stored in the appropriate fields.
    It then sets the imputed counts in the refit adata.

    Note that this function must be run on an object which already contains
    a refit_adata, whose counts, obsm and uns have been set with the
    `set_basic_refit_adata` function.

    Parameters
    ----------
    self : Any
        The object containing the refit adata, the local adata, the replaced
        genes, the trimmed mean normed counts, the size factors, the cooks G
        cutoff, and the replaceable genes.

    """
    assert (
        self.refit_adata is not None
    ), "Refit adata must be loaded before setting the imputed counts."
    assert (
        self.local_adata is not None
    ), "Local adata must be loaded before setting the imputed counts."
    assert (
        "replaced" in self.local_adata.varm.keys()
    ), "Replaced genes must be computed before setting the imputed counts."
    assert (
        "_trimmed_mean_normed_counts" in self.refit_adata.varm.keys()
    ), "Trimmed mean normed counts must be computed before setting the imputed counts."
    assert (
        "size_factors" in self.refit_adata.obsm.keys()
    ), "Size factors must be computed before setting the imputed counts."
    assert (
        "_where_cooks_g_cutoff" in self.local_adata.uns.keys()
    ), "Cooks G cutoff must be computed before setting the imputed counts."
    assert (
        "replaceable" in self.refit_adata.obsm.keys()
    ), "Replaceable genes must be computed before setting the imputed counts."

    trimmed_mean_normed_counts = self.refit_adata.varm["_trimmed_mean_normed_counts"]

    replacement_counts = pd.DataFrame(
        self.refit_adata.obsm["size_factors"][:, None] * trimmed_mean_normed_counts,
        columns=self.refit_adata.var_names,
        index=self.refit_adata.obs_names,
    ).astype(int)

    idx = np.zeros(self.local_adata.shape, dtype=bool)
    idx[self.local_adata.uns["_where_cooks_g_cutoff"]] = True

    # Restrict to the genes to replace
    if "refitted" not in self.local_adata.varm.keys():
        idx = idx[:, self.local_adata.varm["replaced"]]
    else:
        idx = idx[:, self.local_adata.varm["refitted"]]

    # Replace the counts
    self.refit_adata.X[
        self.refit_adata.obsm["replaceable"][:, None] & idx
    ] = replacement_counts.values[self.refit_adata.obsm["replaceable"][:, None] & idx]
