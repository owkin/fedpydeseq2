from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions.get_num_replicates.substeps import (  # noqa: E501
    AggGetCountsLvlForCells,
)
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions.get_num_replicates.substeps import (  # noqa: E501
    LocFinalizeCellCounts,
)
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions.get_num_replicates.substeps import (  # noqa: E501
    LocGetDesignMatrixLevels,
)
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.logging.logging_decorators import log_organisation_method


class GetNumReplicates(
    LocGetDesignMatrixLevels, AggGetCountsLvlForCells, LocFinalizeCellCounts
):
    """Mixin class to get the number of replicates for each combination of factors."""

    @log_organisation_method
    def get_num_replicates(
        self, train_data_nodes, aggregation_node, local_states, round_idx, clean_models
    ):
        """
        Compute the number of replicates for each combination of factors.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: dict
            Local states. Required to propagate intermediate results.

        round_idx: int
            Index of the current round.

        clean_models: bool
            Whether to clean the models after the computation.

        Returns
        -------
        local_states: dict
            Local states, to store the number of replicates and cell level codes.

        round_idx: int
            The updated round index.
        """
        local_states, shared_states, round_idx = local_step(
            local_method=self.loc_get_design_matrix_levels,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Get local matrix design level",
            round_idx=round_idx,
            clean_models=clean_models,
        )
        counts_lvl_share_state, round_idx = aggregation_step(
            aggregation_method=self.agg_get_counts_lvl_for_cells,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Compute counts level",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        local_states, _, round_idx = local_step(
            local_method=self.loc_finalize_cell_counts,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=counts_lvl_share_state,
            aggregation_id=aggregation_node.organization_id,
            description="Finalize cell counts",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        return local_states, round_idx
