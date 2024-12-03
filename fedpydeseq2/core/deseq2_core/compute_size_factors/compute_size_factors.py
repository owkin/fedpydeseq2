"""Module containing the steps for the computation of rough dispersions."""

from fedpydeseq2.core.deseq2_core.compute_size_factors.substeps import AggLogMeans
from fedpydeseq2.core.deseq2_core.compute_size_factors.substeps import (
    LocSetSizeFactorsComputeGramAndFeatures,
)
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step


class ComputeSizeFactors(
    AggLogMeans,
    LocSetSizeFactorsComputeGramAndFeatures,
):
    """Mixin class to implement the computation of size factors.

    Methods
    -------
    compute_size_factors
        The method to compute the size factors, that must be used in the main
        pipeline. It sets the size factors in the local AnnData and computes the
        Gram matrix and feature vector in order to start the next step, i.e.,
        the computation of rough dispersions.

    """

    def compute_size_factors(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        shared_states,
        round_idx,
        clean_models,
    ):
        """Compute size factors.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: dict
            Local states. Required to propagate intermediate results.

        shared_states: list
            Shared states which are the output of the "build_design_matrix" step.
            These shared states contain the following fields:
            - "log_mean" : the log mean of the gene expressions.
            - "n_samples" : the number of samples in each client.

        round_idx: int
            Index of the current round.

        clean_models: bool
            Whether to clean the models after the computation.

        Returns
        -------
        local_states: dict
            Local states. Required to propagate intermediate results.

        shared_states: dict
            Shared states which contain the local information necessary to start
            running the compute rough dispersions step. These shared states contain
            a "local_gram_matrix" and a "local_features" key.

        round_idx: int
            The updated round index.

        """
        # ---- Aggregate means of log gene expressions ----#

        log_mean_aggregated_state, round_idx = aggregation_step(
            aggregation_method=self.aggregate_log_means,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            round_idx=round_idx,
            description="Aggregating local log means",
            clean_models=clean_models,
        )

        # ---- Set local size factors and return next shared states ---- #

        local_states, shared_states, round_idx = local_step(
            local_method=self.local_set_size_factors_compute_gram_and_features,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            round_idx=round_idx,
            input_local_states=local_states,
            input_shared_state=log_mean_aggregated_state,
            aggregation_id=aggregation_node.organization_id,
            description=(
                "Setting local size factors and "
                "computing Gram matrices and feature vectors"
            ),
            clean_models=clean_models,
        )

        return local_states, shared_states, round_idx
