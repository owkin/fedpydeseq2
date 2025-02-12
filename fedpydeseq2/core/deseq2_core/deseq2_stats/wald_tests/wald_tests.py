from fedpydeseq2.core.deseq2_core.deseq2_stats.wald_tests.substeps import (
    AggRunWaldTests,
)
from fedpydeseq2.core.deseq2_core.deseq2_stats.wald_tests.substeps import (
    LocBuildContrastVectorHMatrix,
)
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.logging.logging_decorators import log_organisation_method


class RunWaldTests(LocBuildContrastVectorHMatrix, AggRunWaldTests):
    """Mixin class to implement the computation of the Wald tests.

    Methods
    -------
    run_wald_tests
        The method to compute the Wald tests.
    """

    @log_organisation_method
    def run_wald_tests(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        round_idx,
        clean_models,
    ):
        """Compute the Wald tests.

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
        """
        # --- Build contrast vectors and compute local H matrices --- #
        local_states, shared_states, round_idx = local_step(
            local_method=self.compute_contrast_vector_and_H_matrix,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            round_idx=round_idx,
            input_local_states=local_states,
            input_shared_state=None,  # TODO plug in previous step
            aggregation_id=aggregation_node.organization_id,
            description="Build contrast vectors and compute local H matrices",
            clean_models=clean_models,
        )

        # --- Aggregate the H matrices and run the Wald tests --- #
        wald_shared_state, round_idx = aggregation_step(
            aggregation_method=self.agg_run_wald_tests,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            round_idx=round_idx,
            description="Run Wald tests.",
            clean_models=clean_models,
        )

        return local_states, wald_shared_state, round_idx
