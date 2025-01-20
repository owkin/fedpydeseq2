from loguru import logger

from fedpydeseq2.core.deseq2_core.deseq2_stats.compute_padj import (
    ComputeAdjustedPValues,
)
from fedpydeseq2.core.deseq2_core.deseq2_stats.cooks_filtering import CooksFiltering
from fedpydeseq2.core.deseq2_core.deseq2_stats.wald_tests import RunWaldTests
from fedpydeseq2.core.utils.logging.logging_decorators import log_organisation_method


class DESeq2Stats(RunWaldTests, CooksFiltering, ComputeAdjustedPValues):
    """Mixin class to compute statistics with DESeq2.

    This class encapsulates the Wald tests, the Cooks filtering and the computation
    of adjusted p-values.

    Methods
    -------
    run_deseq2_stats
        Run the DESeq2 statistics pipeline.
        Performs Wald tests, Cook's filtering and computes adjusted p-values.

    """

    cooks_filter: bool

    @log_organisation_method
    def run_deseq2_stats(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        round_idx,
        clean_models,
    ):
        """
        Run the DESeq2 statistics pipeline.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: list[dict]
            Local states. Required to propagate intermediate results.

        round_idx: int
            Index of the current round.

        clean_models: bool
            Whether to clean the models after the computation.


        Returns
        -------
        local_states: dict
            Local states.

        round_idx: int
            The updated round index.

        """
        #### Perform Wald tests ####
        logger.info("Running Wald tests.")

        local_states, wald_shared_state, round_idx = self.run_wald_tests(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=clean_models,
        )

        logger.info("Finished running Wald tests.")

        if self.cooks_filter:
            logger.info("Running Cook's filtering...")
            local_states, wald_shared_state, round_idx = self.cooks_filtering(
                train_data_nodes,
                aggregation_node,
                local_states,
                wald_shared_state,
                round_idx,
                clean_models=clean_models,
            )
            logger.info("Finished running Cook's filtering.")
            logger.info("Computing adjusted p-values...")
        (
            local_states,
            round_idx,
        ) = self.compute_adjusted_p_values(
            train_data_nodes,
            aggregation_node,
            local_states,
            wald_shared_state,
            round_idx,
            clean_models=clean_models,
        )
        logger.info("Finished computing adjusted p-values.")

        return local_states, round_idx
