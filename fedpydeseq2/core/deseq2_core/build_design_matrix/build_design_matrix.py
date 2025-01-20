from fedpydeseq2.core.deseq2_core.build_design_matrix.substeps import (
    AggMergeDesignColumnsBuildContrast,
)
from fedpydeseq2.core.deseq2_core.build_design_matrix.substeps import (
    AggMergeDesignLevels,
)
from fedpydeseq2.core.deseq2_core.build_design_matrix.substeps import LocGetLocalFactors
from fedpydeseq2.core.deseq2_core.build_design_matrix.substeps import (
    LocOderDesignComputeLogMean,
)
from fedpydeseq2.core.deseq2_core.build_design_matrix.substeps import LocSetLocalDesign
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.logging.logging_decorators import log_organisation_method


class BuildDesignMatrix(
    AggMergeDesignColumnsBuildContrast,
    AggMergeDesignLevels,
    LocGetLocalFactors,
    LocSetLocalDesign,
    LocOderDesignComputeLogMean,
):
    """Mixin class to implement the computation of the design matrix.

    Methods
    -------
    build_design_matrix
        The method to build the design matrix, that must be used in the main
        pipeline.

    check_design_matrix
        The method to check the design matrix, that must be used in the main
        pipeline while we are testing.
    """

    @log_organisation_method
    def build_design_matrix(
        self, train_data_nodes, aggregation_node, local_states, round_idx, clean_models
    ):
        """Build the design matrix.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: dict
            Local states. Required to propagate intermediate results.

        round_idx:
            The current round

        clean_models: bool
            Whether to clean the models after the computation.

        Returns
        -------
        local_states: dict
            Local states. Required to propagate intermediate results.

        shared_states: dict
            Shared states containing the necessary local information to start
            the next step of the pipeline, which is computing the size factors.
            They contain a "log_means" key and a "n_samples" key.

        round_idx: int
            The updated round

        """
        # ---- For each design factor, get the list of each center's levels ---- #
        if len(local_states) == 0:
            # In that case, there is no reference dds, and this is the first step of
            # The pipeline
            input_local_states = None
        else:
            # In this case, there was already a step before, and we need to propagate
            # the local states
            input_local_states = local_states

        local_states, shared_states, round_idx = local_step(
            local_method=self.get_local_factors,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            round_idx=round_idx,
            input_local_states=input_local_states,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Computing local design factor levels",
            clean_models=clean_models,
        )

        # ---- For each design factor, merge the list of unique levels ---- #

        design_levels_aggregated_state, round_idx = aggregation_step(
            aggregation_method=self.merge_design_levels,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            round_idx=round_idx,
            description="Merging design levels",
            clean_models=clean_models,
        )

        # ---- Initialize design matrices in each center ---- #

        local_states, shared_states, round_idx = local_step(
            local_method=self.set_local_design,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            round_idx=round_idx,
            input_local_states=local_states,
            input_shared_state=design_levels_aggregated_state,
            aggregation_id=aggregation_node.organization_id,
            description="Setting local design matrices",
            clean_models=clean_models,
        )

        # ---- Merge design columns ---- #

        design_columns_aggregated_state, round_idx = aggregation_step(
            aggregation_method=self.merge_design_columns_and_build_contrast,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            round_idx=round_idx,
            description="Merge local design matrix columns",
            clean_models=clean_models,
        )

        local_states, shared_states, round_idx = local_step(
            local_method=self.order_design_cols_compute_local_log_mean,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            round_idx=round_idx,
            input_shared_state=design_columns_aggregated_state,
            aggregation_id=aggregation_node.organization_id,
            description="Computing local log means",
            clean_models=clean_models,
        )

        return local_states, shared_states, round_idx
