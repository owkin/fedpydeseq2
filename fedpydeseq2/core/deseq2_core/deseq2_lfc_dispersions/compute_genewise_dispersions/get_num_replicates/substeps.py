import numpy as np
import pandas as pd
from anndata import AnnData
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data


class LocGetDesignMatrixLevels:
    """Mixin to get the unique values of the local design matrix."""

    local_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def loc_get_design_matrix_levels(self, data_from_opener, shared_state=dict) -> dict:
        """Get the values of the local design matrix.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.
        shared_state : dict
            Not used.

        Returns
        -------
        dict
            Dictionary with the following key:
            - unique_counts: unique values and counts of the local design matrix
        """
        unique_counts = self.local_adata.obsm["design_matrix"].value_counts()

        return {"unique_counts": unique_counts}


class AggGetCountsLvlForCells:
    """Mixin that aggregate the counts of the design matrix values."""

    @remote
    @log_remote
    def agg_get_counts_lvl_for_cells(self, shared_states: list[dict]) -> dict:
        """Aggregate the counts of the design matrix values.

        Parameters
        ----------
        shared_states : list(dict)
            List of shared states with the following key:
            - unique_counts: unique values and counts of the local design matrix

        Returns
        -------
        dict
            Dictionary with keys labeling the different values taken by the
            overall design matrix. Each values of the dictionary contains the
            sum of the counts of the corresponding design matrix value and the level.
        """
        concat_unique_cont = pd.concat(
            [shared_state["unique_counts"] for shared_state in shared_states], axis=1
        )
        counts_by_lvl = concat_unique_cont.fillna(0).sum(axis=1).astype(int)

        return {"counts_by_lvl": counts_by_lvl}


class LocFinalizeCellCounts:
    """Mixin that finalize the cell counts."""

    local_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def loc_finalize_cell_counts(self, data_from_opener, shared_state=dict) -> None:
        """Finalize the cell counts.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict
            Dictionary with keys labeling the different values taken by the
            overall design matrix. Each values of the dictionary contains the
            sum of the counts of the corresponding design matrix value and the level.
        """
        counts_by_lvl = shared_state["counts_by_lvl"]

        # In order to keep the same objects 'num_replicates' and 'cells' used in
        # PyDESeq2, we provide names (0, 1, 2...) to the possible values of the
        # design matrix, called "lvl".
        # The index of 'num_replicates' is the lvl names (0,1,2...) and its values
        # the counts of these lvl
        # 'cells' index is the index of the cells in the adata and its values the lvl
        # name (0,1,2..) of the cell.
        self.local_adata.uns["num_replicates"] = pd.Series(counts_by_lvl.values)
        self.local_adata.obs["cells"] = [
            np.argwhere(counts_by_lvl.index == tuple(design))[0, 0]
            for design in self.local_adata.obsm["design_matrix"].values
        ]
