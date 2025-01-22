"""Module containing the substeps for the computation of size factors."""


import anndata as ad
import numpy as np
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.utils.aggregation import aggregate_means
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data


class AggLogMeans:
    """Mixin to compute the global mean given the local results."""

    @remote
    @log_remote
    def aggregate_log_means(self, shared_states):
        """Compute the global mean given the local results.

        Parameters
        ----------
        shared_states : list
            List of results (local_mean, n_samples) from training nodes.

        Returns
        -------
        dict
            Global mean of log counts, and new all-zero genes if in refit mode.
        """
        tot_mean = aggregate_means(
            [state["log_mean"] for state in shared_states],
            [state["n_samples"] for state in shared_states],
        )

        return {"global_log_mean": tot_mean}


class LocSetSizeFactorsComputeGramAndFeatures:
    """Mixin to set local size factors and return local Gram matrices and features.

    This Mixin implements the method to perform the transition between the
    compute_size_factors and compute_rough_dispersions steps. It sets the size
    factors in the local AnnData and computes the Gram matrix and feature vector.

    Methods
    -------
    local_set_size_factors_compute_gram_and_features
        The method to set the size factors and compute the Gram matrix and feature.
    """

    local_adata: ad.AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def local_set_size_factors_compute_gram_and_features(
        self,
        data_from_opener,
        shared_state,
    ) -> dict:
        # pylint: disable=unused-argument
        """Set local size factor and compute Gram matrix and feature vector.

        This is a local method, used to fit the rough dispersions.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict
            Shared state containing the "global_log_mean" key.

        Returns
        -------
        dict
            Local gram matrices and feature vectors to be shared via shared_state to
            the aggregation node.
        """
        #### ---- Compute size factors ---- ####

        global_log_means = shared_state["global_log_mean"]
        # Filter out genes with -∞ log means
        filtered_genes = ~np.isinf(global_log_means)

        log_ratios = (
            np.log(self.local_adata.X[:, filtered_genes])
            - global_log_means[filtered_genes]
        )
        # Compute sample-wise median of log ratios
        log_medians = np.median(log_ratios, axis=1)
        # Return raw counts divided by size factors (exponential of log ratios)
        # and size factors
        self.local_adata.obsm["size_factors"] = np.exp(log_medians)
        self.local_adata.layers["normed_counts"] = (
            self.local_adata.X / self.local_adata.obsm["size_factors"][:, None]
        )

        design = self.local_adata.obsm["design_matrix"].values

        return {
            "local_gram_matrix": design.T @ design,
            "local_features": design.T @ self.local_adata.layers["normed_counts"],
        }
