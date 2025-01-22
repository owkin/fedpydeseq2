import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import f  # type: ignore
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.utils.layers import prepare_cooks_agg
from fedpydeseq2.core.utils.layers import prepare_cooks_local
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data


class LocFindCooksOutliers:
    """Mixin class to find the local cooks outliers.

    Attributes
    ----------
    local_adata : AnnData
        Local AnnData object.
        Is expected to have a "tot_num_samples" key in uns.

    refit_cooks : bool
        Whether to refit the cooks outliers.


    Methods
    -------
    find_local_cooks_outliers
        Find the local cooks outliers.
    """

    local_adata: AnnData
    refit_cooks: bool

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    @prepare_cooks_local
    def find_local_cooks_outliers(
        self,
        data_from_opener,
        shared_state: dict,
    ) -> dict:
        """Find the local cooks outliers.

        This method is expected to run on the results of the Wald tests.

        Parameters
        ----------
        data_from_opener : AnnData
            Not used.

        shared_state : dict
            Shared state from the previous step with the following
            keys:
            - p_values: np.ndarray of shape (n_genes,)
            - wald_statistics: np.ndarray of shape (n_genes,)
            - wald_se: np.ndarray of shape (n_genes,)

        Returns
        -------
        shared_state : dict
            A shared state with the following fields:
            - local_cooks_outliers: np.ndarray of shape (n_genes,)
                It is a boolean array indicating whether a gene is a cooks outlier.
            - cooks_cutoff: float
                The cutoff used to define the fact that a gene is a cooks outlier.
        """
        # Save these in the local adata
        self.local_adata.varm["p_values"] = shared_state["p_values"]
        self.local_adata.varm["wald_statistics"] = shared_state["wald_statistics"]
        self.local_adata.varm["wald_se"] = shared_state["wald_se"]

        tot_num_samples = self.local_adata.uns["tot_num_samples"]
        num_vars = self.local_adata.uns["n_params"]

        cooks_cutoff = f.ppf(0.99, num_vars, tot_num_samples - num_vars)

        # Take into account whether we already replaced outliers
        cooks_layer = (
            "replace_cooks"
            if self.refit_cooks and self.local_adata.varm["refitted"].sum() > 0
            else "cooks"
        )

        if cooks_layer == "replace_cooks":
            assert "replaced" in self.local_adata.varm.keys()
            replace_cooks = pd.DataFrame(self.local_adata.layers["cooks"].copy())
            replace_cooks.loc[
                self.local_adata.obsm["replaceable"], self.local_adata.varm["refitted"]
            ] = 0.0
            self.local_adata.layers["replace_cooks"] = replace_cooks

        use_for_max = self.local_adata.obs["cells"].apply(
            lambda x: (self.local_adata.uns["num_replicates"] >= 3).loc[x]
        )

        cooks_outliers = (
            (self.local_adata[use_for_max, :].layers[cooks_layer] > cooks_cutoff)
            .any(axis=0)
            .copy()
        )

        return {"local_cooks_outliers": cooks_outliers, "cooks_cutoff": cooks_cutoff}


class AggregateCooksOutliers:
    """Mixin class to aggregate the cooks outliers.

    Methods
    -------
    agg_cooks_outliers
        Aggregate the local cooks outliers.
    """

    @remote
    @log_remote
    @prepare_cooks_agg
    def agg_cooks_outliers(self, shared_states: list[dict]) -> dict:
        """Aggregate the local cooks outliers.

        Parameters
        ----------
        shared_states : list[dict]
            List of shared states from the local step with the following keys:
            - local_cooks_outliers: np.ndarray of shape (n_genes,)
            - cooks_cutoff: float

        Returns
        -------
        shared_state : dict
            Aggregated cooks outliers.
            It is a dictionary with the following fields:
            - cooks_outliers: np.ndarray of shape (n_genes,)
                It is a boolean array indicating whether a gene is a cooks outlier in
                any of the local datasets
            - cooks_cutoff: float
                The cutoff used to define the fact that a gene is a cooks outlier.
        """
        return {
            "cooks_outliers": np.any(
                [state["local_cooks_outliers"] for state in shared_states], axis=0
            ),
            "cooks_cutoff": shared_states[0]["cooks_cutoff"],
        }


class LocGetMaxCooks:
    """Mixin class to get the maximum cooks distance for the outliers.

    Attributes
    ----------
    local_adata : AnnData
        Local AnnData object.

    Methods
    -------
    get_max_local_cooks
        Get the maximum cooks distance for the outliers.
    """

    local_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def get_max_local_cooks(
        self,
        data_from_opener,
        shared_state: dict,
    ) -> dict:
        """Get the maximum cooks distance for the outliers.

        Parameters
        ----------
        data_from_opener : AnnData
            Not used.

        shared_state : dict
            Shared state from the previous step with the following
            keys:
            - cooks_outliers: np.ndarray of shape (n_genes,)
                It is a boolean array indicating whether a gene is a cooks outlier.
            - cooks_cutoff: float

        Returns
        -------
        shared_state : dict
            A shared state with the following fields:
            - local_max_cooks: np.ndarray of shape (n_cooks_genes,)
                The maximum cooks distance for the outliers in the local dataset.
            - cooks_outliers: np.ndarray of shape (n_genes,)
                It is a boolean array indicating whether a gene is a cooks outlier.
        """
        cooks_outliers = shared_state["cooks_outliers"]
        cooks_cutoff = shared_state["cooks_cutoff"]

        max_cooks = np.max(self.local_adata.layers["cooks"][:, cooks_outliers], axis=0)

        max_cooks[max_cooks <= cooks_cutoff] = 0.0

        max_cooks_idx = self.local_adata.layers["cooks"][:, cooks_outliers].argmax(
            axis=0
        )

        max_cooks_value = self.local_adata.layers["cooks"][:, cooks_outliers][
            max_cooks_idx, np.arange(len(max_cooks))
        ]

        max_cooks_gene_counts = self.local_adata.X[:, cooks_outliers][
            max_cooks_idx, np.arange(len(max_cooks))
        ]

        # Save the max cooks gene counts and max cooks value
        self.local_adata.uns["max_cooks_gene_counts"] = max_cooks_gene_counts
        self.local_adata.uns["max_cooks_value"] = max_cooks_value

        return {
            "local_max_cooks": max_cooks,
            "cooks_outliers": cooks_outliers,
        }


class AggMaxCooks:
    """Mixin class to aggregate the max cooks distances.

    Methods
    -------
    agg_max_cooks
        Aggregate the local max cooks distances.
    """

    @remote
    @log_remote
    def agg_max_cooks(self, shared_states: list[dict]) -> dict:
        """Aggregate the local max cooks.

        Parameters
        ----------
        shared_states : list[dict]
            List of shared states from the local step with the following keys:
            - local_max_cooks: np.ndarray of shape (n_genes,)
                The local maximum cooks distance for the outliers.
            - cooks_outliers: np.ndarray of shape (n_genes,)
                It is a boolean array indicating whether a gene is a cooks outlier.

        Returns
        -------
        shared_state : dict
            Aggregated max cooks.
            It is a dictionary with the following fields:
            - max_cooks: np.ndarray of shape (n_cooks_genes,)
                The maximum cooks distance for the outliers in the aggregated dataset.
            - cooks_outliers: np.ndarray of shape (n_genes,)
                It is a boolean array indicating whether a gene is a cooks outlier.
        """
        return {
            "max_cooks": np.max(
                [state["local_max_cooks"] for state in shared_states], axis=0
            ),
            "cooks_outliers": shared_states[0]["cooks_outliers"],
        }


class LocGetMaxCooksCounts:
    """Mixin class to get the maximum cooks counts for the outliers.

    Attributes
    ----------
    local_adata : AnnData
        Local AnnData object.

    Methods
    -------
    get_max_local_cooks_gene_counts
        Get the maximum cooks counts for the outliers.
    """

    local_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def get_max_local_cooks_gene_counts(
        self,
        data_from_opener,
        shared_state: dict,
    ) -> dict:
        """Get the maximum cooks counts for the outliers.

        Parameters
        ----------
        data_from_opener : AnnData
            Not used.

        shared_state : dict
            Shared state from the previous step with the following
            keys:
            - max_cooks: np.ndarray of shape (n_cooks_genes,)
                The maximum cooks distance for the outliers.
            - cooks_outliers: np.ndarray of shape (n_genes,)
                It is a boolean array indicating whether a gene is a cooks outlier.

        Returns
        -------
        shared_state : dict
            A shared state with the following fields:
            - local_max_cooks_gene_counts: np.ndarray of shape (n_cooks_genes,)
                For each gene, the array contains the gene counts corresponding to the
                maximum cooks distance for that gene if the maximum cooks distance
                in the local dataset is equal to the maximum cooks distance in the
                aggregated dataset, and nan otherwise.
            - cooks_outliers: np.ndarray of shape (n_genes,)
                It is a boolean array indicating whether a gene is a cooks outlier.
        """
        max_cooks = shared_state["max_cooks"]
        cooks_outliers = shared_state["cooks_outliers"]

        max_cooks_gene_counts = self.local_adata.uns["max_cooks_gene_counts"].copy()
        max_cooks_value = self.local_adata.uns["max_cooks_value"].copy()

        # Remove them from the uns field as they are no longer needed
        del self.local_adata.uns["max_cooks_gene_counts"]
        del self.local_adata.uns["max_cooks_value"]

        max_cooks_gene_counts[
            max_cooks_value < max_cooks
        ] = -1  # We can use a < because the count value are non negative integers.

        max_cooks_gene_counts_ma = np.ma.masked_array(
            max_cooks_gene_counts, max_cooks_gene_counts == -1
        )

        return {
            "local_max_cooks_gene_counts": max_cooks_gene_counts_ma,
            "cooks_outliers": cooks_outliers,
        }


class AggMaxCooksCounts:
    """Mixin class to aggregate the max cooks gene counts.

    Methods
    -------
    agg_max_cooks_gene_counts
        Aggregate the local max cooks gene counts. The goal is to have the gene
        counts corresponding to the maximum cooks distance for each gene across
        all datasets.
    """

    @remote
    @log_remote
    def agg_max_cooks_gene_counts(self, shared_states: list[dict]) -> dict:
        """Aggregate the local max cooks gene counts.

        Parameters
        ----------
        shared_states : list[dict]
            List of shared states from the local step with the following keys:
            - local_max_cooks_gene_counts: np.ndarray of shape (n_genes,)
                The local maximum cooks gene counts for the outliers.
            - cooks_outliers: np.ndarray of shape (n_genes,)
                It is a boolean array indicating whether a gene is a cooks outlier.

        Returns
        -------
        shared_state : dict
            A shared state with the following fields:
            - max_cooks_gene_counts: np.ndarray of shape (n_cooks_genes,)
                For each gene, the array contains the gene counts corresponding to the
                maximum cooks distance for that gene across all datasets.
            - cooks_outliers: np.ndarray of shape (n_genes,)
                It is a boolean array indicating whether a gene is a cooks outlier.
        """
        return {
            "max_cooks_gene_counts": np.ma.stack(
                [state["local_max_cooks_gene_counts"] for state in shared_states],
                axis=0,
            ).min(axis=0),
            "cooks_outliers": shared_states[0]["cooks_outliers"],
        }


class LocCountNumberSamplesAbove:
    """Mixin class to count the number of samples above the max cooks gene counts.

    Attributes
    ----------
    local_adata : AnnData

    Methods
    -------
    count_local_number_samples_above
        Count the number of samples above the max cooks gene counts.
    """

    local_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def count_local_number_samples_above(
        self,
        data_from_opener,
        shared_state: dict,
    ) -> dict:
        """Count the number of samples above the max cooks gene counts.

        Parameters
        ----------
        data_from_opener : AnnData
            Not used.

        shared_state : dict
            Shared state from the previous step with the following
            keys:
            - cooks_outliers: np.ndarray of shape (n_genes,)
                It is a boolean array indicating whether a gene is a cooks outlier.
            - max_cooks_gene_counts: np.ndarray of shape (n_genes,)
                For each gene, the array contains the gene counts corresponding to the
                maximum cooks distance for that gene across all datasets.

        Returns
        -------
        shared_state : dict
            A shared state with the following fields:
            - local_num_samples_above: np.ndarray of shape (n_cooks_genes,)
                For each gene, the array contains the number of samples above the
                maximum cooks gene counts.
            - cooks_outliers: np.ndarray of shape (n_genes,)
                It is a boolean array indicating whether a gene is a cooks outlier.
            - p_values: np.ndarray of shape (n_genes,)
                The p-values from the Wald test.
            - wald_statistic: np.ndarray of shape (n_genes,)
                The Wald statistics from the Wald test.
            - wald_se: np.ndarray of shape (n_genes,)
                The Wald standard errors from the Wald test.
        """
        cooks_outliers = shared_state["cooks_outliers"]
        max_cooks_gene_counts = shared_state["max_cooks_gene_counts"]

        num_samples_above = np.sum(
            self.local_adata.X[:, cooks_outliers] > max_cooks_gene_counts, axis=0
        )

        return {
            "local_num_samples_above": num_samples_above,
            "cooks_outliers": cooks_outliers,
            "p_values": self.local_adata.varm["p_values"],
            "wald_statistics": self.local_adata.varm["wald_statistics"],
            "wald_se": self.local_adata.varm["wald_se"],
        }


class AggCooksFiltering:
    """Mixin class to aggregate the cooks filtering.

    Methods
    -------
    agg_cooks_filtering
        Aggregate the local number of samples above.
    """

    @remote
    @log_remote
    def agg_cooks_filtering(self, shared_states: list[dict]) -> dict:
        """Aggregate the local number of samples above to get cooks filtered genes.

        Parameters
        ----------
        shared_states : list[dict]
            List of shared states from the local step with the following keys:
            - local_num_samples_above: np.ndarray of shape (n_genes,)
                The local number of samples above the max cooks gene counts.
            - cooks_outliers: np.ndarray of shape (n_genes,)
                It is a boolean array indicating whether a gene is a cooks outlier.
            - p_values: np.ndarray of shape (n_genes,)
                The p-values from the Wald test.
            - wald_statistics: np.ndarray of shape (n_genes,)
                The Wald statistics from the Wald test.
            - wald_se: np.ndarray of shape (n_genes,)
                The Wald standard errors from the Wald test.

        Returns
        -------
        dict
            A shared state with the following fields:
            - p_values: np.ndarray of shape (n_genes,)
                The p-values from the Wald test with nan for the cooks outliers.
            - wald_statistics: np.ndarray of shape (n_genes,)
                The Wald statistics.
            - wald_se: np.ndarray of shape (n_genes,)
                The Wald standard errors.
        """
        # Find the number of samples with counts above the max cooks
        cooks_outliers = shared_states[0]["cooks_outliers"]

        num_samples_above_max_cooks = np.sum(
            [state["local_num_samples_above"] for state in shared_states], axis=0
        )

        # If that number is greater than 3, set the cooks filter to false
        cooks_outliers[cooks_outliers] = num_samples_above_max_cooks < 3

        # Set the p-values to nan on cooks outliers
        p_values = shared_states[0]["p_values"]
        p_values[cooks_outliers] = np.nan

        return {
            "p_values": p_values,
            "wald_statistics": shared_states[0]["wald_statistics"],
            "wald_se": shared_states[0]["wald_se"],
        }
