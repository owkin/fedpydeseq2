from loguru import logger
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef

from fedpydeseq2.core.deseq2_core.build_design_matrix import BuildDesignMatrix
from fedpydeseq2.core.deseq2_core.compute_cook_distance import ComputeCookDistances
from fedpydeseq2.core.deseq2_core.compute_size_factors import ComputeSizeFactors
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions import DESeq2LFCDispersions
from fedpydeseq2.core.deseq2_core.deseq2_stats.deseq2_stats import DESeq2Stats
from fedpydeseq2.core.deseq2_core.replace_outliers import ReplaceCooksOutliers
from fedpydeseq2.core.deseq2_core.replace_refitted_values import ReplaceRefittedValues
from fedpydeseq2.core.deseq2_core.save_pipeline_results import SavePipelineResults


class DESeq2FullPipe(
    BuildDesignMatrix,
    ComputeSizeFactors,
    DESeq2LFCDispersions,
    ComputeCookDistances,
    ReplaceCooksOutliers,
    ReplaceRefittedValues,
    DESeq2Stats,
    SavePipelineResults,
):
    """A Mixin class to run the full DESeq2 pipeline.

    Methods
    -------
    run_deseq_pipe
        The method to run the full DESeq2 pipeline.
    """

    def run_deseq_pipe(
        self,
        train_data_nodes: list[TrainDataNode],
        aggregation_node: AggregationNode,
        local_states: dict[str, LocalStateRef],
        round_idx: int = 0,
        clean_models: bool = True,
        clean_last_model: bool = False,
    ):
        """Run the DESeq2 pipeline.

        Parameters
        ----------
        train_data_nodes : list[TrainDataNode]
            List of the train nodes.
        aggregation_node : AggregationNode
            Aggregation node.
        local_states : dict[str, LocalStateRef]
            Local states.
        round_idx : int
            Round index.
        clean_models : bool
            Whether to clean the models after the computation. (default: ``True``).
            Note that as intermediate steps are very memory consuming, it is recommended
            to clean the models after each step.
        clean_last_model : bool
            Whether to clean the last model. (default: ``False``).
        """
        #### Build design matrices ####

        logger.info("Building design matrices...")

        local_states, log_mean_shared_states, round_idx = self.build_design_matrix(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=clean_models,
        )

        logger.info("Finished building design matrices.")

        #### Compute size factors ####
        # Note: in refit mode, this doesn't recompute size factors,
        # just the log features

        logger.info("Computing size factors...")

        (
            local_states,
            gram_features_shared_states,
            round_idx,
        ) = self.compute_size_factors(
            train_data_nodes,
            aggregation_node,
            local_states,
            shared_states=log_mean_shared_states,
            round_idx=round_idx,
            clean_models=clean_models,
        )

        logger.info("Finished computing size factors.")

        #### Compute LFC and dispersions ####

        logger.info("Running LFC and dispersions.")

        local_states, round_idx = self.run_deseq2_lfc_dispersions(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            gram_features_shared_states=gram_features_shared_states,
            round_idx=round_idx,
            clean_models=clean_models,
        )

        logger.info("Finished running LFC and dispersions.")

        logger.info("Computing Cook distances...")

        (
            local_states,
            cooks_shared_state,
            round_idx,
        ) = self.compute_cook_distance(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=clean_models,
        )

        logger.info("Finished computing Cook distances.")

        #### Refit cooks if necessary ####
        if self.refit_cooks:
            logger.info("Refitting Cook outliers...")
            (
                local_states,
                gram_features_shared_states,
                round_idx,
            ) = self.replace_outliers(
                train_data_nodes,
                aggregation_node,
                local_states,
                cooks_shared_state,
                round_idx,
                clean_models=clean_models,
            )

            local_states, round_idx = self.run_deseq2_lfc_dispersions(
                train_data_nodes=train_data_nodes,
                aggregation_node=aggregation_node,
                local_states=local_states,
                gram_features_shared_states=gram_features_shared_states,
                round_idx=round_idx,
                clean_models=clean_models,
                refit_mode=True,
            )
            # Replace values in the main ``local_adata`` object
            local_states, round_idx = self.replace_refitted_values(
                train_data_nodes=train_data_nodes,
                aggregation_node=aggregation_node,
                local_states=local_states,
                round_idx=round_idx,
                clean_models=clean_models,
            )

            logger.info("Finished refitting Cook outliers.")

        #### Compute DESeq2 statistics ####

        logger.info("Running DESeq2 statistics.")

        local_states, round_idx = self.run_deseq2_stats(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=clean_models,
        )

        logger.info("Finished running DESeq2 statistics.")

        # Build the results that will be downloaded at the end of the pipeline.

        logger.info("Saving pipeline results.")
        self.save_pipeline_results(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=clean_models,
        )

        logger.info("Finished saving pipeline results.")
