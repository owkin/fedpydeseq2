from fedpydeseq2.core.deseq2_core.compute_cook_distance.substeps import (
    AggComputeDispersionForCook,
)
from fedpydeseq2.core.deseq2_core.compute_cook_distance.substeps import (
    LocComputeSqerror,
)
from fedpydeseq2.core.deseq2_core.compute_cook_distance.substeps import (
    LocGetNormedCounts,
)
from fedpydeseq2.core.fed_algorithms import ComputeTrimmedMean
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.logging.logging_decorators import log_organisation_method


class ComputeCookDistances(
    ComputeTrimmedMean,
    LocComputeSqerror,
    LocGetNormedCounts,
    AggComputeDispersionForCook,
):
    """Mixin class to compute Cook's distances.

    Methods
    -------
    compute_cook_distance
        The method to compute Cook's distances.

    """

    trimmed_mean_num_iter: int

    @log_organisation_method
    def compute_cook_distance(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        round_idx,
        clean_models,
    ):
        """
        Compute Cook's distances.

        Parameters
        ----------
        train_data_nodes: list
            list of TrainDataNode.

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
            Local states. The new local state contains Cook's distances.

        dispersion_for_cook_shared_state: dict
            Shared state with the dispersion values for Cook's distances, in a
            "cooks_dispersions" key.

        round_idx: int
            The updated round index.

        """
        local_states, agg_shared_state, round_idx = self.compute_trim_mean(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=clean_models,
            layer_used="normed_counts",
            mode="cooks",
            trim_ratio=None,
            n_iter=self.trimmed_mean_num_iter,
        )

        local_states, shared_states, round_idx = local_step(
            local_method=self.local_compute_sqerror,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=agg_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Compute local sqerror",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        local_states, agg_shared_state, round_idx = self.compute_trim_mean(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=clean_models,
            layer_used="sqerror",
            mode="cooks",
            trim_ratio=None,
            n_iter=self.trimmed_mean_num_iter,
        )

        local_states, shared_states, round_idx = local_step(
            local_method=self.local_get_normed_count_means,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=agg_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Get normed count means",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        dispersion_for_cook_shared_state, round_idx = aggregation_step(
            aggregation_method=self.agg_compute_dispersion_for_cook,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Compute dispersion for Cook distances",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        return local_states, dispersion_for_cook_shared_state, round_idx
