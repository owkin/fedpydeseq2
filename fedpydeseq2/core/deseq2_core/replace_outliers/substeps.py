import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import f  # type: ignore
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.layers.build_layers import set_normed_counts
from fedpydeseq2.core.utils.layers.build_refit_adata import set_basic_refit_adata
from fedpydeseq2.core.utils.layers.build_refit_adata import (
    set_imputed_counts_refit_adata,
)
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data


class LocFindCooksOutliers:
    """Find local Cooks outliers."""

    local_adata: AnnData
    min_replicates: int

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def loc_find_cooks_outliers(
        self,
        data_from_opener,
        shared_state: dict,
    ) -> dict:
        """
        Find local Cooks outliers by comparing the cooks distance to a threshold.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict, optional
            Not used.

        Returns
        -------
        dict
            Shared state containing:
            - "local_genes_to_replace": genes with Cook's distance above the threshold,
            - "replaceable_samples": a boolean indicating whether there is at least one
               sample with enough replicates to replace it.

        """
        # Find replaceable samples
        n_or_more = self.local_adata.uns["num_replicates"] >= self.min_replicates

        self.local_adata.obsm["replaceable"] = n_or_more[
            self.local_adata.obs["cells"]
        ].values

        # Find genes with Cook's distance above the threshold
        n_params = self.local_adata.uns["n_params"]
        cooks_cutoff = f.ppf(
            0.99, n_params, self.local_adata.uns["tot_num_samples"] - n_params
        )

        self.local_adata.uns["_where_cooks_g_cutoff"] = np.where(
            self.local_adata.layers["cooks"] > cooks_cutoff
        )

        local_idx_to_replace = (self.local_adata.layers["cooks"] > cooks_cutoff).any(
            axis=0
        )
        local_genes_to_replace = self.local_adata.var_names[local_idx_to_replace]

        return {
            "local_genes_to_replace": set(local_genes_to_replace),
            "replaceable_samples": self.local_adata.obsm["replaceable"].any(),
        }


class AggMergeOutlierGenes:
    """Build the global list of genes to replace."""

    @remote
    @log_remote
    def agg_merge_outlier_genes(
        self,
        shared_states: list[dict],
    ) -> dict:
        """
        Merge the lists of genes to replace.

        Parameters
        ----------
        shared_states : list
            List of dictionaries containing:
            - "local_genes_to_replace": genes with Cook's distance above the threshold,
            - "replaceable_samples": a boolean indicating whether there is at least
               one sample with enough replicates to replace it.

        Returns
        -------
        dict
            A dictionary with a unique key: "genes_to_replace" containing the list
            of genes for which to replace outlier values.
        """
        # If no sample is replaceable, we can skip
        any_replaceable = any(state["replaceable_samples"] for state in shared_states)

        if not any_replaceable:
            return {"genes_to_replace": set()}

        else:
            # Take the union of all local list of genes to replace
            genes_to_replace = set.union(
                *[state["local_genes_to_replace"] for state in shared_states]
            )

            return {
                "genes_to_replace": genes_to_replace,
            }


class LocSetRefitAdata:
    """Mixin to replace cooks outliers locally."""

    local_adata: AnnData
    refit_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def loc_set_refit_adata(
        self,
        data_from_opener,
        shared_state: dict,
    ) -> None:
        """
        Set a refit adata containing the counts of the genes to replace.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict
            A dictionary with a "genes_to_replace" key, containing the list of genes
            for which to replace outlier values.
        """
        # Save the information on which genes will be replaced
        genes_to_replace = pd.Series(False, index=self.local_adata.var_names)
        genes_to_replace[list(shared_state["genes_to_replace"])] = True
        self.local_adata.varm["replaced"] = genes_to_replace.values

        # Copy the values corresponding to the genes to refit in the refit_adata
        set_basic_refit_adata(self)


class LocReplaceCooksOutliers:
    """Mixin to replace cooks outliers locally."""

    local_adata: AnnData
    refit_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def loc_replace_cooks_outliers(
        self,
        data_from_opener,
        shared_state: dict,
    ) -> dict:
        """
        Replace outlier counts with imputed values.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict
            A dictionary with a "trimmed_mean_normed_counts" key, containing the
            trimmed means to use to compute the imputed values.

        Returns
        -------
        dict
            A dictionary containing:
            - "loc_new_all_zero": a boolean array indicating which genes are now
              all-zero.
        """
        # Set the trimmed mean normed counts in the varm
        self.refit_adata.varm["_trimmed_mean_normed_counts"] = shared_state[
            "trimmed_mean_normed_counts"
        ]

        set_imputed_counts_refit_adata(self)

        # Find new all-zero columns
        new_all_zeroes = self.refit_adata.X.sum(axis=0) == 0

        # Return the new local logmeans
        with np.errstate(divide="ignore"):  # ignore division by zero warnings
            return {
                "loc_new_all_zeroes": new_all_zeroes,
            }


class AggNewAllZeros:
    """Mixin to compute the new all zeros and share to the centers."""

    @remote
    @log_remote
    def aggregate_new_all_zeros(self, shared_states: list) -> dict:
        """Compute the global mean given the local results.

        Parameters
        ----------
        shared_states : list
            List of results (local_mean, n_samples) from training nodes.
            In refit mode, also contains "loc_new_all_zero".

        Returns
        -------
        dict
            New all-zero genes.
        """
        # Find genes that are all zero due to imputation of counts
        new_all_zeroes = np.all(
            [state["loc_new_all_zeroes"] for state in shared_states], axis=0
        )

        return {"new_all_zeroes": new_all_zeroes}


class LocSetNewAllZerosAndGetFeatures:
    """Mixin to set the new all zeros and return local features.

    This Mixin implements the method to perform the transition towards the
    compute_rough_dispersions steps after refitting. It sets the new all zeros
    genes in the local AnnData and computes the local features to be shared
    to the aggregation node.

    Methods
    -------
    local_set_new_all_zeros_get_features
        The method to set the new all zeros genes and compute the local features.

    """

    local_adata: ad.AnnData
    refit_adata: ad.AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def local_set_new_all_zeros_get_features(
        self,
        data_from_opener,
        shared_state,
    ) -> dict:
        """
        Set the new_all_zeros field and get the features.

        This method is used to set the new_all_zeros field in the local_adata uns
        field. This is the set of genes that are all zero after outlier replacement.

        It then restricts the refit_adata to the genes which are not all_zero.

        Finally, it computes the local features to be shared via shared_state to the
        aggregation node.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict
            Shared state containing the "new_all_zeroes" key.

        Returns
        -------
        dict
            Local feature vector to be shared via shared_state to
            the aggregation node.
        """
        # Take all-zero genes into account
        new_all_zeroes = shared_state["new_all_zeroes"]

        self.local_adata.uns["new_all_zeroes_genes"] = self.refit_adata.var_names[
            new_all_zeroes
        ]

        self.local_adata.varm["refitted"] = self.local_adata.varm["replaced"].copy()
        # Only replace if genes are not all zeroes after outlier replacement
        self.local_adata.varm["refitted"][
            self.local_adata.varm["refitted"]
        ] = ~new_all_zeroes

        # RESTRICT REFIT ADATA TO NOT NEW ALL ZEROES
        self.refit_adata = self.refit_adata[:, ~new_all_zeroes].copy()

        # Update normed counts
        set_normed_counts(self.refit_adata)

        #### ---- Compute Gram matrix and feature vector ---- ####

        design = self.refit_adata.obsm["design_matrix"].values

        return {
            "local_features": design.T @ self.refit_adata.layers["normed_counts"],
        }
