from fedpydeseq2.core.deseq2_core.deseq2_stats.compute_padj.substeps import (
    IndependentFiltering,
)
from fedpydeseq2.core.deseq2_core.deseq2_stats.compute_padj.substeps import (
    PValueAdjustment,
)
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.logging.logging_decorators import log_organisation_method


class ComputeAdjustedPValues(IndependentFiltering, PValueAdjustment):
    """Mixin class to implement the computation of adjusted p-values.

    Attributes
    ----------
    independent_filter: bool
        A boolean flag to indicate whether to use independent filtering or not.

    Methods
    -------
    compute_adjusted_p_values
        A method to compute adjusted p-values.
        Runs independent filtering if self.independent_filter is True.
        Runs BH method otherwise.

    """

    independent_filter: bool = False

    @log_organisation_method
    def compute_adjusted_p_values(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        wald_test_shared_state,
        round_idx,
        clean_models,
    ):
        """Compute adjusted p-values.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: dict
            Local states. Required to propagate intermediate results.

        wald_test_shared_state: dict
            Shared states containing the Wald test results.

        round_idx: int
            The current round.

        clean_models: bool
            If True, the models are cleaned.

        Returns
        -------
        local_states: dict
            Local states. Required to propagate intermediate results.


        round_idx: int
            The updated round index.

        """
        if self.independent_filter:
            local_states, _, round_idx = local_step(
                local_method=self.run_independent_filtering,
                train_data_nodes=train_data_nodes,
                output_local_states=local_states,
                round_idx=round_idx,
                input_local_states=local_states,
                input_shared_state=wald_test_shared_state,
                aggregation_id=aggregation_node.organization_id,
                description="Compute adjusted P values using independent filtering.",
                clean_models=clean_models,
            )
        else:
            local_states, _, round_idx = local_step(
                local_method=self.run_p_value_adjustment,
                train_data_nodes=train_data_nodes,
                output_local_states=local_states,
                round_idx=round_idx,
                input_local_states=local_states,
                input_shared_state=wald_test_shared_state,
                aggregation_id=aggregation_node.organization_id,
                description="Compute adjusted P values using BH method.",
                clean_models=clean_models,
            )

        return local_states, round_idx
