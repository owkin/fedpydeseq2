"""Module to implement the step to set the local reference for the DESeq2Strategy."""
import pickle as pkl
from pathlib import Path
from typing import Any

import anndata as ad
from pydeseq2.dds import DeseqDataSet
from substrafl.remote import remote_data

from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote_data


class SetLocalReference:
    """Mixin to set the local reference for the DESeq2Strategy."""

    reference_data_path: str | Path | None
    local_reference_dds: DeseqDataSet | None
    reference_dds_name: str

    def set_local_reference_dataset(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        round_idx,
        clean_models,
    ):
        """Set the local reference DeseqDataset.

        This function restricts the global reference DeseqDataset (assumed to be
        constructed from the
        pooled data and save in the right subdirectory, see
        `fedpydeseq2.data.tcga_setup` module
        for more details) to the data in the local dataset.

        It then sets the local reference DeseqDataset as an attribute of the strategy.

        Parameters
        ----------
        train_data_nodes : list
            List of TrainDataNode.
        aggregation_node : AggregationNode
            Aggregation Node.
        local_states : dict
            Local states. Required to propagate intermediate results.
        round_idx : int
            Index of the current round.
        clean_models : bool
            Whether to clean the models after the computation.

        Returns
        -------
        local_states : dict
            Local states. Required to propagate intermediate results.
        shared_states : Any
            Empty shared state.
        round_idx : int
            Index of the current round.
        """
        local_states, shared_states, round_idx = local_step(
            local_method=self.set_local_reference_dataset_remote,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=None,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Setting the reference local dds object, "
            "if given a reference_data_path.",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        return local_states, shared_states, round_idx

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def set_local_reference_dataset_remote(
        self, data_from_opener: ad.AnnData, shared_state: Any
    ):
        """Set the local reference DeseqDataset.

        This function restricts the global reference DeseqDataset (assumed to be
        constructed from the
        pooled data and save in the right subdirectory, see
        `fedpydeseq2.data.tcga_setup` module
        for more details) to the data in the local dataset.

        It then sets the local reference DeseqDataset as an attribute of the strategy.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            The local AnnData. Used here to access the center_id.
        shared_state : Any
            Not used here.
        """
        if self.reference_data_path is not None:
            reference_data_path = Path(self.reference_data_path).resolve()
            center_id = data_from_opener.uns["center_id"]
            # Get the
            path = (
                reference_data_path
                / f"center_{center_id}"
                / f"{self.reference_dds_name}.pkl"
            )
            with open(path, "rb") as file:
                local_reference_dds = pkl.load(file)
            self.local_reference_dds = local_reference_dds
