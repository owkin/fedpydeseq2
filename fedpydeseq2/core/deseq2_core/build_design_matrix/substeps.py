"""Module containing the substeps for the computation of design matrices.

This module contains all these substeps as mixin classes.
"""

import anndata as ad
import numpy as np
import pandas as pd
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.utils import build_contrast
from fedpydeseq2.core.utils import build_design_matrix
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data


class AggMergeDesignColumnsBuildContrast:
    """Mixin to merge the columns of the design matrices and build contrast."""

    design_factors: list[str]
    continuous_factors: list[str] | None
    contrast: list[str] | None

    @remote
    @log_remote
    def merge_design_columns_and_build_contrast(self, shared_states):
        """Merge the columns of the design matrices and build constrasts.

        Parameters
        ----------
        shared_states : list
            List of results (dictionaries of design columns) from training nodes.

        Returns
        -------
        dict
            Shared state containing:
            - merged_columns: the names of the columns that the local design matrices
              should have.
            - contrast: the contrast (in a list of strings form) to be used for the
              DESeq2 model.
        """
        merged_columns = pd.Index([])

        for state in shared_states:
            merged_columns = merged_columns.union(state["design_columns"])

        # We now also have everything to compute the contrasts
        contrast = build_contrast(
            self.design_factors,
            merged_columns,
            self.continuous_factors,
            self.contrast,
        )

        return {"merged_columns": merged_columns, "contrast": contrast}


class AggMergeDesignLevels:
    """Mixin to merge the levels of the design factors."""

    categorical_factors: list[str]

    @remote
    @log_remote
    def merge_design_levels(self, shared_states):
        """Merge the levels of the design factors.

        Parameters
        ----------
        shared_states : list
            List of results (dictionaries of local_levels) from training nodes.

        Returns
        -------
        dict
            Dictionary of unique levels for each factor.
        """
        # merge levels
        merged_levels = {factor: set() for factor in self.categorical_factors}
        for factor in self.categorical_factors:
            for state in shared_states:
                merged_levels[factor] = set(state["local_levels"][factor]).union(
                    merged_levels[factor]
                )

        return {
            "merged_levels": {
                factor: np.array(list(levels))
                for factor, levels in merged_levels.items()
            }
        }


class LocGetLocalFactors:
    """Mixin to get the list of unique levels for each categorical design factor."""

    categorical_factors: list[str]

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def get_local_factors(
        self, data_from_opener, shared_state=None
    ):  # pylint: disable=unused-argument
        """Get the list of unique levels for each categorical design factor.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Copied in local anndata objects.

        shared_state : None, optional
            Not used.

        Returns
        -------
        dict
            A dictionary of unique local levels for each factor.
        """
        self.local_adata = data_from_opener.copy()
        return {
            "local_levels": {
                factor: self.local_adata.obs[factor].unique()
                for factor in self.categorical_factors
            }
        }


class LocSetLocalDesign:
    """Mixin to set the design matrices in centers."""

    local_adata: ad.AnnData
    design_factors: list[str]
    continuous_factors: list[str] | None
    ref_levels: dict[str, str] | None

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def set_local_design(
        self,
        data_from_opener,
        shared_state,
    ):
        # pylint: disable=unused-argument
        """Set the design matrices in centers.

        Returns their columns in order to harmonize them.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict
            Shared state with a "design_columns" key containing a dictionary with, for
            each design factor, the names of its unique levels.

        Returns
        -------
        dict
            Local design columns.
        """
        self.local_adata.obsm["design_matrix"] = build_design_matrix(
            metadata=self.local_adata.obs,
            design_factors=self.design_factors,
            continuous_factors=self.continuous_factors,
            levels=shared_state["merged_levels"],
            ref_levels=self.ref_levels,
        )
        return {"design_columns": self.local_adata.obsm["design_matrix"].columns}


class LocOderDesignComputeLogMean:
    """Mixin to order design cols and compute the local log mean.

    Attributes
    ----------
    local_adata : ad.AnnData
        The local AnnData.

    Methods
    -------
    order_design_cols_compute_local_log_mean
        Order design columns and compute the local log mean.
    """

    local_adata: ad.AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def order_design_cols_compute_local_log_mean(
        self, data_from_opener, shared_state=None
    ):
        """Order design columns and compute the local log mean.

        This function also sets the contrast in the local AnnData,
        and saves the number of parameters in the uns field.


        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict
            Shared state with:
            - "merged_columns" a set containing the names of columns that the design
                matrix should have.
            - "contrast" the contrast to be used for the DESeq2 model.

        Returns
        -------
        dict
            Local mean of logs and number of samples.
        """
        #### ----Step 1: Order design columns---- ####

        self.local_adata.uns["contrast"] = shared_state["contrast"]

        for col in shared_state["merged_columns"]:
            if col not in self.local_adata.obsm["design_matrix"].columns:
                self.local_adata.obsm["design_matrix"][col] = 0

        # Reorder columns for consistency
        self.local_adata.obsm["design_matrix"] = self.local_adata.obsm["design_matrix"][
            shared_state["merged_columns"]
        ]

        # Save the number of params in an uns field for easy access
        self.local_adata.uns["n_params"] = self.local_adata.obsm["design_matrix"].shape[
            1
        ]

        #### ----Step 2: Compute local log mean---- ####

        with np.errstate(divide="ignore"):  # ignore division by zero warnings
            return {
                "log_mean": np.log(data_from_opener.X).mean(axis=0),
                "n_samples": data_from_opener.n_obs,
            }
