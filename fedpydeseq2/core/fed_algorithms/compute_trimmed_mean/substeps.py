"""Module to implement the substeps for comuting the trimmed mean.

This module contains all these substeps as mixin classes.
"""
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.fed_algorithms.compute_trimmed_mean.utils import get_scale
from fedpydeseq2.core.fed_algorithms.compute_trimmed_mean.utils import get_trim_ratio
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data


class LocInitTrimmedMean:
    """Mixin class to implement the local initialisation of the trimmed mean algo."""

    local_adata: AnnData
    refit_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def loc_init_trimmed_mean(
        self,
        data_from_opener,
        shared_state,
        layer_used: str,
        mode: Literal["normal", "cooks"] = "normal",
        refit: bool = False,
        min_replicates_trimmed_mean: int = 3,
    ) -> dict:
        """
        Initialise the trimmed mean algo, by providing the lower and max bounds.

        Parameters
        ----------
        data_from_opener : AnnData
            Unused, all the necessary info is stored in the local adata.

        shared_state : dict
            Not used, all the necessary info is stored in the local adata.

        layer_used : str
            Name of the layer used to compute the trimmed mean.

        mode : Literal["normal", "cooks"]
            Mode of the trimmed mean algo. If "cooks", the function will be applied
            either on the normalized counts or the squared error.
            It will be applied per level, except if there are not enough samples.

        refit : bool
            If true, the function will use the refit adata to compute the trimmed mean.

        min_replicates_trimmed_mean : int
            Minimum number of replicates to compute the trimmed mean.

        Returns
        -------
        dict
            If mode is "normal" or if mode is "cooks" and there are not enough samples,
            to compute the trimmed mean per level, a dictionary with the following keys
                - max_values: np.ndarray of size (n_genes,)
                - min_values: np.ndarray of size (n_genes,)
                - use_lvl: False
            otherwise, a dictionary with the max_values and min_values keys, nested
            inside a dictionary with the levels as keys, plus a use_lvl with value True
        """
        if refit:
            adata = self.refit_adata
        else:
            adata = self.local_adata

        if mode == "cooks":
            # Check that the layer is either cooks or normed counts
            assert layer_used in ["sqerror", "normed_counts"]
            # Check that num replicates is in the uns
            assert "num_replicates" in adata.uns, "No num_replicates in the adata"
            use_lvl = adata.uns["num_replicates"].max() >= min_replicates_trimmed_mean
            assert "cells" in adata.obs, "No cells column in the adata"

        else:
            use_lvl = False
        result = {"use_lvl": use_lvl}
        if use_lvl:
            # In that case, we know we are in cooks mode
            admissible_levels = adata.uns["num_replicates"][
                adata.uns["num_replicates"] >= min_replicates_trimmed_mean
            ].index

            shared_state = {lvl: shared_state for lvl in admissible_levels}
            for lvl in admissible_levels:
                mask = adata.obs["cells"] == lvl
                result[lvl] = self.loc_init_trimmed_mean_per_lvl(  # type: ignore
                    data_from_opener, shared_state[lvl], layer_used, mask, refit
                )
            return result
        else:
            result.update(
                self.loc_init_trimmed_mean_per_lvl(
                    data_from_opener,
                    shared_state,
                    layer_used,
                    mask=np.ones(adata.n_obs, dtype=bool),
                    refit=refit,
                )
            )
            return result

    def loc_init_trimmed_mean_per_lvl(
        self,
        data_from_opener,
        shared_state,
        layer_used: str,
        mask,
        refit: bool = False,
    ) -> dict:
        """
        Initialise the trimmed mean algo, by providing the lower and max bounds.

        Parameters
        ----------
        data_from_opener : AnnData
            Unused, all the necessary info is stored in the local adata.

        shared_state : dict
            Not used, all the necessary info is stored in the local adata.

        layer_used : str
            Name of the layer used to compute the trimmed mean.

        mask : np.ndarray
            Mask to filter values used in the min and max computation.

        refit : bool
            If true, the function will use the refit adata to compute the trimmed mean.

        Returns
        -------
        dict
            Dictionary with the following keys
                - max_values: np.ndarray of size (n_genes,)
                - min_values: np.ndarray of size (n_genes,)
                - n_samples: int, number of samples
        """
        if refit:
            adata = self.refit_adata
        else:
            adata = self.local_adata

        assert layer_used in adata.layers
        local_adata_filtered = adata[mask]
        if local_adata_filtered.n_obs > 0:
            max_values = local_adata_filtered.layers[layer_used].max(axis=0)
            min_values = local_adata_filtered.layers[layer_used].min(axis=0)
        else:
            max_values = np.zeros(adata.n_vars) * np.nan
            min_values = np.zeros(adata.n_vars) * np.nan
        return {
            "max_values": max_values,
            "min_values": min_values,
        }


class AggInitTrimmedMean:
    """Mixin class for the aggregation of the init of the trimmed mean algo."""

    @remote
    @log_remote
    def agg_init_trimmed_mean(
        self,
        shared_states: list[dict],
    ) -> dict:
        """
        Compute the initial global upper and lower bounds.

        Parameters
        ----------
        shared_states : list[dict]
            If use_lvl is False (in any shared state),
            list of dictionaries with the following keys:
                - max_values: np.ndarray of size (n_genes,)
                - min_values: np.ndarray of size (n_genes,)
            If use_lvl is True, list of dictionaries with the same keys as above
            nested inside a dictionary with the levels as keys.

        Returns
        -------
        dict
            use_level is a key present in all input shared states, and will be passed
            on to the output shared state
            If use_lvl is False, dict with the following keys:
               - upper_bounds_thresholds : np.ndarray of size (n_genes, 2)
               - lower_bounds_thresholds : np.ndarray of size (n_genes, 2)
            otherwise, a dictionary with the same keys for nested inside a dictionary
            with the levels as keys.
        """
        use_lvl = shared_states[0]["use_lvl"]
        result = {"use_lvl": use_lvl}
        if use_lvl:
            for lvl in shared_states[0].keys():
                if lvl == "use_lvl":
                    continue
                result[lvl] = self.agg_init_trimmed_mean_per_lvl(
                    [state[lvl] for state in shared_states]
                )
            return result
        else:
            result.update(self.agg_init_trimmed_mean_per_lvl(shared_states))
            return result

    def agg_init_trimmed_mean_per_lvl(self, shared_states: list[dict]) -> dict:
        """
        Compute the initial global upper and lower bounds.

        Parameters
        ----------
        shared_states : list[dict]
            List of dictionaries with the following keys:
                - max_values: np.ndarray of size (n_genes,)
                - min_values: np.ndarray of size (n_genes,)

        Returns
        -------
        dict
            dict with the following keys:
               - upper_bounds_thresholds : np.ndarray of size (n_genes, 2)
               - lower_bounds_thresholds : np.ndarray of size (n_genes, 2)
        """
        # To initialize the dichotomic search of the quantile thresholds, we need to
        # set the upper and lower bounds of the thresholds.
        upper_bounds_thresholds = np.nanmax(
            np.array([state["max_values"] for state in shared_states]), axis=0
        )
        lower_bounds_thresholds = np.nanmin(
            np.array([state["min_values"] for state in shared_states]), axis=0
        )

        # We are looking for two thresholds, one for the upper quantile and one for the
        # lower quantile. We initialize the search with the same value for both.
        upper_bounds_thresholds = np.vstack([upper_bounds_thresholds] * 2).T
        lower_bounds_thresholds = np.vstack([lower_bounds_thresholds] * 2).T

        upper_bounds_thresholds = upper_bounds_thresholds.astype(np.float32)
        lower_bounds_thresholds = lower_bounds_thresholds.astype(np.float32)

        return {
            "upper_bounds_thresholds": upper_bounds_thresholds,
            "lower_bounds_thresholds": lower_bounds_thresholds,
        }


class LocalIterationTrimmedMean:
    """Mixin class to implement the local iteration of the trimmed mean algo."""

    local_adata: AnnData
    refit_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def local_iteration_trimmed_mean(
        self,
        data_from_opener,
        shared_state,
        layer_used: str,
        mode: Literal["normal", "cooks"] = "normal",
        trim_ratio: float | None = None,
        refit: bool = False,
    ) -> dict:
        """
        Local iteration of the trimmed mean algo.

        Parameters
        ----------
        data_from_opener : AnnData
            Not used, all the necessary info is stored in the local adata.

        shared_state : dict
            Dictionary with the following keys:
               - upper_bounds_thresholds : np.ndarray of size (n_genes,2). Not used.
               - lower_bounds_thresholds : np.ndarray of size (n_genes,2). Not used.
            If use_lvl is true, the dictionary is nested with the levels as keys.

        layer_used : str
            Name of the layer used to compute the trimmed mean.

        mode : Literal["normal", "cooks"]
            Mode of the trimmed mean algo. If "cooks", the function will be applied
            either on the normalized counts or the squared error.
            It will be applied per level, except if there are not enough samples.
            Moreover, trim ratios will be computed based on the number of replicates.
            If "normal", the function will be applied on the whole dataset, using the
            trim_ratio parameter.

        trim_ratio : float or None
            Ratio of the samples to be trimmed. Must be between 0 and 0.5. Must be
            None if mode is "cooks", and float if mode is "normal".

        refit : bool
            If true, the function will use the refit adata to compute the trimmed mean.

        Returns
        -------
        dict
            Dictionary containing the following keys:
                - num_strictly_above: np.ndarray[int] of size (n_genes,2)
                - upper_bounds_thresholds: np.ndarray of size (n_genes,2)
                - lower_bounds_thresholds: np.ndarray of size (n_genes,2)
                - n_samples: int
                - trim_ratio: float
            If use_lvl is true, the dictionary is nested with the levels as keys.

        """
        if refit:
            adata = self.refit_adata
        else:
            adata = self.local_adata
        use_lvl = shared_state["use_lvl"]
        result = {"use_lvl": use_lvl}

        if mode == "cooks":
            assert trim_ratio is None
        else:
            assert trim_ratio is not None

        if mode == "cooks" and use_lvl:
            for lvl in shared_state.keys():
                if lvl == "use_lvl":
                    continue
                mask = adata.obs["cells"] == lvl
                result[lvl] = self.local_iteration_trimmed_mean_per_lvl(
                    data_from_opener, shared_state[lvl], layer_used, mask, refit
                )
                trim_ratio = get_trim_ratio(adata.uns["num_replicates"][lvl])
                result[lvl]["trim_ratio"] = trim_ratio
            return result

        result.update(
            self.local_iteration_trimmed_mean_per_lvl(
                data_from_opener,
                shared_state,
                layer_used,
                mask=np.ones(adata.n_obs, dtype=bool),
                refit=refit,
            )
        )
        if mode == "cooks":
            result["trim_ratio"] = 0.125
        else:
            result["trim_ratio"] = trim_ratio

        return result

    def local_iteration_trimmed_mean_per_lvl(
        self, data_from_opener, shared_state, layer_used, mask, refit: bool = False
    ) -> dict:
        """
        Local iteration of the trimmed mean algo.

        Parameters
        ----------
        data_from_opener : AnnData
            Not used, all the necessary info is stored in the local adata.

        shared_state : dict
            Dictionary with the following keys:
               - upper_bounds_thresholds : np.ndarray of size (n_genes,2). Not used.
               - lower_bounds_thresholds : np.ndarray of size (n_genes,2). Not used.

        layer_used : str
            Name of the layer used to compute the trimmed mean.

        mask : np.ndarray
            Mask to filter values used in the quantile computation.

        refit : bool
            If true, the function will use the refit adata to compute the trimmed mean.

        Returns
        -------
        dict
            Dictionary containing the following keys:
                - num_strictly_above: np.ndarray[int] of size (n_genes,2)
                - upper_bounds_thresholds: np.ndarray of size (n_genes,2)
                - lower_bounds_thresholds: np.ndarray of size (n_genes,2)
                - n_samples: int

        """
        if refit:
            adata = self.refit_adata
        else:
            adata = self.local_adata

        # We don't need to pass the thresholds in the share states as it's always the
        # mean of the upper and lower bounds.
        threshold = (
            shared_state["upper_bounds_thresholds"]
            + shared_state["lower_bounds_thresholds"]
        ) / 2
        local_adata_filtered = adata[mask]
        # Array of size (n_genes, 2) containing the number of samples above the
        # thresholds.
        num_strictly_above = (
            local_adata_filtered.layers[layer_used][..., None] > threshold[None, ...]
        ).sum(axis=0)

        return {
            "num_strictly_above": num_strictly_above,
            "upper_bounds_thresholds": shared_state["upper_bounds_thresholds"],
            "lower_bounds_thresholds": shared_state["lower_bounds_thresholds"],
            "n_samples": local_adata_filtered.n_obs,
        }


class AggIterationTrimmedMean:
    """Mixin class of the aggregation of the iteration of the trimmed mean algo."""

    @remote
    @log_remote
    def agg_iteration_trimmed_mean(
        self,
        shared_states: list[dict],
    ) -> dict:
        """
        Compute the initial global upper and lower bounds.

        Parameters
        ----------
        shared_states : list[dict]
            List of dictionnaries with the following keys:
                - num_strictly_above: np.ndarray[int] of size (n_genes,2)
                - upper_bounds_thresholds: np.ndarray of size (n_genes,2)
                - lower_bounds_thresholds: np.ndarray of size (n_genes,2)
                - n_samples: int
                - trim_ratio: float
            If use_lvl is true, the dictionary is nested with the levels as keys.

        Returns
        -------
        dict
            Dictionary with the following keys:
               - upper_bounds_thresholds : np.ndarray of size (n_genes, 2)
               - lower_bounds_thresholds : np.ndarray of size (n_genes, 2)
            If use_lvl is true, the dictionary is nested with the levels as keys.
        """
        use_lvl = shared_states[0]["use_lvl"]
        result = {"use_lvl": use_lvl}
        if use_lvl:
            for lvl in shared_states[0].keys():
                if lvl == "use_lvl":
                    continue
                result[lvl] = self.agg_iteration_trimmed_mean_per_lvl(
                    [state[lvl] for state in shared_states]
                )
            return result
        else:
            result.update(self.agg_iteration_trimmed_mean_per_lvl(shared_states))
            return result

    def agg_iteration_trimmed_mean_per_lvl(
        self,
        shared_states: list[dict],
    ) -> dict:
        """
        Aggregate step of the iteration of the trimmed mean algo.

        Parameters
        ----------
        shared_states : list[dict]
            List of dictionary containing the following keys:
                - num_strictly_above: np.ndarray[int] of size (n_genes,2)
                - upper_bounds_thresholds: np.ndarray of size (n_genes,2)
                - lower_bounds_thresholds: np.ndarray of size (n_genes,2)
                - n_samples: int
                - trim_ratio: float

        Returns
        -------
        dict
            Dictionary with the following keys:
               - upper_bounds_thresholds : np.ndarray of size (n_genes,2)
               - lower_bounds_thresholds : np.ndarray of size (n_genes,2)
            If use_lvl is true, the dictionary is nested with the levels as keys.
        """
        trim_ratio = shared_states[0]["trim_ratio"]
        upper_bounds_thresholds = shared_states[0]["upper_bounds_thresholds"]
        lower_bounds_thresholds = shared_states[0]["lower_bounds_thresholds"]

        n_samples = np.sum([state["n_samples"] for state in shared_states])

        n_trim = np.floor(n_samples * trim_ratio)
        # Targets contain the number of samples we want to have above the two
        # thresholds.

        targets = np.array([n_trim, n_samples - n_trim])

        # We sum the number of samples above the thresholds for each gene.
        agg_n_samples_strictly_above_quantiles = np.sum(
            [state["num_strictly_above"] for state in shared_states],
            axis=0,
        )

        # Mask of size (n_genes,2) indicating for each gene and each of the two
        # thresholds if the number of samples above the threshold is too high.
        mask_threshold_too_high = (
            agg_n_samples_strictly_above_quantiles < targets[None, :]
        )

        # Similarly, we create a mask for the case where the number of samples above the
        # thresholds is too low.
        mask_threshold_too_low = (
            agg_n_samples_strictly_above_quantiles > targets[None, :]
        )

        ## Update the thresholds and bounds when the thresholds are two high or too low.
        upper_bounds_thresholds[mask_threshold_too_high] = (
            upper_bounds_thresholds[mask_threshold_too_high]
            + lower_bounds_thresholds[mask_threshold_too_high]
        ) / 2.0

        lower_bounds_thresholds[mask_threshold_too_low] = (
            upper_bounds_thresholds[mask_threshold_too_low]
            + lower_bounds_thresholds[mask_threshold_too_low]
        ) / 2.0

        return {
            "upper_bounds_thresholds": upper_bounds_thresholds,
            "lower_bounds_thresholds": lower_bounds_thresholds,
        }


class LocFinalTrimmedMean:
    """Mixin class to implement the local finalisation of the trimmed mean algo."""

    local_adata: AnnData
    refit_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def final_local_trimmed_mean(
        self,
        data_from_opener,
        shared_state,
        layer_used: str,
        mode: Literal["normal", "cooks"] = "normal",
        trim_ratio: float | None = None,
        refit: bool = False,
    ) -> dict:
        """
        Finalise the trimmed mean algo by computing the trimmed mean.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            Unused, all the necessary info is stored in the local adata.

        shared_state : dict
            Dictionary with the following keys:
               - upper_bounds_thresholds : np.ndarray of size (n_genes,2). Not  used
               - lower_bounds_thresholds : np.ndarray of size (n_genes,2). Not used
            If use_lvl is true, the dictionary is nested with the levels as keys.

        layer_used : str
            Name of the layer used to compute the trimmed mean.

        mode : Literal["normal", "cooks"]
            Mode of the trimmed mean algo. If "cooks", the function will be applied
            either on the normalized counts or the squared error.
            It will be applied per level, except if there are not enough samples.
            Moreover, trim ratios will be computed based on the number of replicates.
            If "normal", the function will be applied on the whole dataset, using the
            trim_ratio parameter.

        trim_ratio : float or None
            Ratio of the samples to be trimmed. Must be between 0 and 0.5. Must be
            None if mode is "cooks", and float if mode is "normal".


        refit : bool
            If true, the function will use the refit adata to compute the trimmed mean.

        Returns
        -------
        dict
            Dictionary with the following keys:
            - trimmed_local_sum : np.ndarray(float) of size (n_genes,2)
            - n_samples : np.ndarray(int) of size (n_genes,2)
            - num_strictly_above : np.ndarray(int) of size (n_genes,2)
            - upper_bounds_thresholds : np.ndarray of size (n_genes,2)
            - lower_bounds_thresholds : np.ndarray of size (n_genes,2)
            If use_lvl is true, the dictionary is nested with the levels as keys.

        """
        if refit:
            adata = self.refit_adata
        else:
            adata = self.local_adata

        use_lvl = shared_state["use_lvl"]
        result = {"use_lvl": use_lvl}
        if mode == "cooks" and use_lvl:
            for lvl in shared_state.keys():
                if lvl == "use_lvl":
                    continue
                mask = adata.obs["cells"] == lvl
                result[lvl] = self.final_local_trimmed_mean_per_lvl(
                    data_from_opener, shared_state[lvl], layer_used, mask, refit
                )
                trim_ratio = get_trim_ratio(adata.uns["num_replicates"][lvl])
                if layer_used == "sqerror":
                    scale = get_scale(adata.uns["num_replicates"][lvl])
                    result[lvl]["scale"] = scale
                result[lvl]["trim_ratio"] = trim_ratio
            return result
        result.update(
            self.final_local_trimmed_mean_per_lvl(
                data_from_opener,
                shared_state,
                layer_used,
                mask=np.ones(adata.n_obs, dtype=bool),
                refit=refit,
            )
        )
        if mode == "cooks":
            assert trim_ratio is None
            result["trim_ratio"] = 0.125
            if layer_used == "sqerror":
                result["scale"] = 1.51
        else:
            assert trim_ratio is not None
            result["trim_ratio"] = trim_ratio
        return result

    def final_local_trimmed_mean_per_lvl(
        self,
        data_from_opener,
        shared_state,
        layer_used,
        mask,
        refit: bool = False,
    ) -> dict:
        """
        Finalise the trimmed mean algo by computing the trimmed mean.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            Unused, all the necessary info is stored in the local adata.

        shared_state : dict
            Dictionary with the following keys:
               - upper_bounds_thresholds : np.ndarray of size (n_genes,2). Not  used
               - lower_bounds_thresholds : np.ndarray of size (n_genes,2). Not used

        layer_used : str
            Name of the layer used to compute the trimmed mean.

        mask : np.ndarray
            Mask to filter values used in the quantile computation.

        refit : bool
            If true, the function will use the refit adata to compute the trimmed mean.

        Returns
        -------
        dict
            Dictionary with the following keys:
            - trimmed_local_sum : np.ndarray(float) of size (n_genes,2)
            - n_samples : np.ndarray(int) of size (n_genes,2)
            - num_strictly_above : np.ndarray(int) of size (n_genes,2)
            - upper_bounds_thresholds : np.ndarray of size (n_genes,2)
            - lower_bounds_thresholds : np.ndarray of size (n_genes,2)

        """
        if refit:
            adata = self.refit_adata
        else:
            adata = self.local_adata

        # we create an explicit copy to avoid ImplicitModificationWarning
        local_adata_filtered = adata[mask].copy()
        current_thresholds = (
            shared_state["upper_bounds_thresholds"]
            + shared_state["lower_bounds_thresholds"]
        ) / 2.0

        num_strictly_above = (
            local_adata_filtered.layers[layer_used][..., None]
            > current_thresholds[None, ...]
        ).sum(axis=0)

        mask_upper_threshold = (
            local_adata_filtered.layers[layer_used]
            > current_thresholds[..., 0][None, :]
        )
        mask_lower_threshold = (
            local_adata_filtered.layers[layer_used]
            <= current_thresholds[..., 1][None, :]
        )
        local_adata_filtered.layers[
            f"trimmed_{layer_used}"
        ] = local_adata_filtered.layers[layer_used].copy()
        local_adata_filtered.layers[f"trimmed_{layer_used}"][
            mask_upper_threshold | mask_lower_threshold
        ] = 0

        return {
            "trimmed_local_sum": local_adata_filtered.layers[
                f"trimmed_{layer_used}"
            ].sum(axis=0),
            "n_samples": local_adata_filtered.n_obs,
            "num_strictly_above": num_strictly_above,
            "upper_bounds_thresholds": shared_state["upper_bounds_thresholds"],
            "lower_bounds_thresholds": shared_state["lower_bounds_thresholds"],
        }


class AggFinalTrimmedMean:
    """Mixin class of the aggregation of the finalisation of the trimmed mean algo."""

    @remote
    @log_remote
    def final_agg_trimmed_mean(
        self,
        shared_states: list[dict],
        layer_used: str,
        mode: Literal["normal", "cooks"] = "normal",
    ) -> dict:
        """
        Compute the initial global upper and lower bounds.

        Parameters
        ----------
        shared_states : list[dict]
            List of dictionnaries with the following keys:
            - trimmed_local_sum : np.ndarray(float) of size (n_genes,2)
            - n_samples : np.ndarray(int) of size (n_genes,2)
            - num_strictly_above : np.ndarray(int) of size (n_genes,2)
            - upper_bounds_thresholds : np.ndarray of size (n_genes,2)
            - lower_bounds_thresholds : np.ndarray of size (n_genes,2)
            If use_lvl is true, the dictionary is nested with the levels as keys.

        layer_used : str
            Name of the layer used to compute the trimmed mean.

        mode : Literal["normal", "cooks"]
            Mode of the trimmed mean algo. If "cooks", the function will be applied
            either on the normalized counts or the squared error.
            It will be applied per level, except if there are not enough samples.
            Moreover, trim ratios will be computed based on the number of replicates.
            If "normal", the function will be applied on the whole dataset, using the
            trim_ratio parameter.


        Returns
        -------
        dict
            If mode is "cooks" and if the layer is "sqerror", a dictionary with the
            "varEst" key containing
                - The maximum of the trimmed means per level if use_level is true,
                rescaled by a scale factor depending on the number of replicates
                - The trimmed mean of the whole dataset otherwise
                scaled by 1.51.
            else, if mode is cooks and use_lvl is true, a dictionary with a
            trimmed_mean_normed_counts key containing a dataframe
            with the trimmed means per level, levels being columns
            else, a dictionary with the following keys:
                - trimmed_mean_layer_used : np.ndarray(float) of size (n_genes)


        """
        use_lvl = shared_states[0]["use_lvl"]
        if mode == "cooks" and use_lvl:
            result = {}
            for lvl in shared_states[0].keys():
                if lvl == "use_lvl":
                    continue
                result[lvl] = self.final_agg_trimmed_mean_per_lvl(
                    [state[lvl] for state in shared_states], layer_used
                )[f"trimmed_mean_{layer_used}"]
            if layer_used == "sqerror":
                return {"varEst": pd.DataFrame.from_dict(result).max(axis=1).to_numpy()}
            else:
                return {f"trimmed_mean_{layer_used}": pd.DataFrame.from_dict(result)}
        elif mode == "cooks" and layer_used == "sqerror":
            return {
                "varEst": self.final_agg_trimmed_mean_per_lvl(
                    shared_states, layer_used
                )["trimmed_mean_sqerror"]
            }
        return self.final_agg_trimmed_mean_per_lvl(shared_states, layer_used)

    def final_agg_trimmed_mean_per_lvl(
        self,
        shared_states: list[dict],
        layer_used: str,
    ) -> dict:
        """
        Aggregate step of the finalisation of the trimmed mean algo.

        Parameters
        ----------
        shared_states :  list[dict]
            List of dictionary containing the following keys:
                - trimmed_local_sum : np.ndarray(float) of size (n_genes,2)
                - n_samples : np.ndarray(int) of size (n_genes,2)
                - num_strictly_above : np.ndarray(int) of size (n_genes,2)
                - upper_bounds : np.ndarray of size (n_genes,2)
                - lower_bounds : np.ndarray of size (n_genes,2)

        layer_used : str
            Name of the layer used to compute the trimmed mean.


        Returns
        -------
        dict
            Dictionary with the following keys:
                - trimmed_mean_layer_used : np.ndarray(float) of size (n_genes)

        """
        trim_ratio = shared_states[0]["trim_ratio"]
        n_samples = np.sum([state["n_samples"] for state in shared_states])
        agg_n_samples_strictly_above_quantiles = np.sum(
            [state["num_strictly_above"] for state in shared_states],
            axis=0,
        )
        n_trim = np.floor(n_samples * trim_ratio)
        targets = np.array([n_trim, n_samples - n_trim])
        effective_n_samples = n_samples - 2 * n_trim
        trimmed_sum = np.sum(
            [state["trimmed_local_sum"] for state in shared_states], axis=0
        )
        current_thresholds = (
            shared_states[0]["upper_bounds_thresholds"]
            + shared_states[0]["lower_bounds_thresholds"]
        ) / 2.0

        # The following lines deal with the "tie" cases, i.e. where duplicate values
        # fall on both part of the "n_trimmed" position. In that case,
        # agg_n_samples_strictly_above_quantiles is different from target.
        # "delta_sample_above_quantile" encode how many elements were wrongly
        # trimmed or not trimmed. We know that these elements were close to the
        # values of the threshold up to ~2^{-n_iter} precision. We can then correct the
        # trimmed sum easily using the threshold values.

        delta_sample_above_quantile = (
            agg_n_samples_strictly_above_quantiles - targets[None, :]
        )
        trimmed_sum = (
            trimmed_sum
            + delta_sample_above_quantile[..., 0] * current_thresholds[..., 0]
        )
        trimmed_sum = (
            trimmed_sum
            - delta_sample_above_quantile[..., 1] * current_thresholds[..., 1]
        )
        trimmed_mean = trimmed_sum / effective_n_samples
        if "scale" in shared_states[0].keys():
            scale = shared_states[0]["scale"]
            trimmed_mean = trimmed_mean * scale
        return {f"trimmed_mean_{layer_used}": trimmed_mean}
