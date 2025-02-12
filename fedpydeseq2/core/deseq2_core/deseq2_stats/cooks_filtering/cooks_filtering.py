"""Module to implement the base Mixin class for Cooks filtering."""

from fedpydeseq2.core.deseq2_core.deseq2_stats.cooks_filtering.substeps import (
    AggCooksFiltering,
)
from fedpydeseq2.core.deseq2_core.deseq2_stats.cooks_filtering.substeps import (
    AggMaxCooks,
)
from fedpydeseq2.core.deseq2_core.deseq2_stats.cooks_filtering.substeps import (
    AggMaxCooksCounts,
)
from fedpydeseq2.core.deseq2_core.deseq2_stats.cooks_filtering.substeps import (
    AggregateCooksOutliers,
)
from fedpydeseq2.core.deseq2_core.deseq2_stats.cooks_filtering.substeps import (
    LocCountNumberSamplesAbove,
)
from fedpydeseq2.core.deseq2_core.deseq2_stats.cooks_filtering.substeps import (
    LocFindCooksOutliers,
)
from fedpydeseq2.core.deseq2_core.deseq2_stats.cooks_filtering.substeps import (
    LocGetMaxCooks,
)
from fedpydeseq2.core.deseq2_core.deseq2_stats.cooks_filtering.substeps import (
    LocGetMaxCooksCounts,
)
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.logging.logging_decorators import log_organisation_method


class CooksFiltering(
    LocFindCooksOutliers,
    AggregateCooksOutliers,
    LocGetMaxCooks,
    AggMaxCooks,
    LocGetMaxCooksCounts,
    AggMaxCooksCounts,
    LocCountNumberSamplesAbove,
    AggCooksFiltering,
):
    """A class to perform Cooks filtering of p-values.

    Methods
    -------
    cooks_filtering
        The method to find Cooks outliers.
    """

    @log_organisation_method
    def cooks_filtering(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        wald_test_shared_state,
        round_idx,
        clean_models,
    ):
        """Perform Cooks filtering.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: list[dict]
            Local states. Required to propagate intermediate results.

        wald_test_shared_state : dict
            A shared state containing the Wald test results.
            These results are the following fields:
            - "p_values": p-values of the Wald test.
            - "wald_statistics" : Wald statistics.
            - "wald_se" : Wald standard errors.

        round_idx: int
            Index of the current round.

        clean_models: bool
            Whether to clean the models after the computation.

        Returns
        -------
        local_states: dict
            Local states. The new local state contains Cook's distances.

        shared_state: dict
            A new shared state containing the following fields:
            - "p_values": p-values of the Wald test, updated to be nan for Cook's
            outliers.
            - "wald_statistics" : Wald statistics, for compatibility.
            - "wald_se" : Wald standard errors, for compatibility.

        round_idx: int
            The updated round index.
        """
        local_states, shared_states, round_idx = local_step(
            local_method=self.find_local_cooks_outliers,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=wald_test_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Find local Cook's outliers",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        cooks_outliers_shared_state, round_idx = aggregation_step(
            aggregation_method=self.agg_cooks_outliers,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Find the global Cook's outliers",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        local_states, shared_states, round_idx = local_step(
            local_method=self.get_max_local_cooks,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=cooks_outliers_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Get local max cooks distance",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        max_cooks_distance_shared_state, round_idx = aggregation_step(
            aggregation_method=self.agg_max_cooks,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Get the max cooks distance for the outliers",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        local_states, shared_states, round_idx = local_step(
            local_method=self.get_max_local_cooks_gene_counts,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=max_cooks_distance_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Get the local max gene counts for the Cook's outliers",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        max_cooks_gene_counts_shared_state, round_idx = aggregation_step(
            aggregation_method=self.agg_max_cooks_gene_counts,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Get the max gene counts for the Cook's outliers",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        local_states, shared_states, round_idx = local_step(
            local_method=self.count_local_number_samples_above,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=max_cooks_gene_counts_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Count the number of samples above the max gene counts",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        cooks_filtered_shared_state, round_idx = aggregation_step(
            aggregation_method=self.agg_cooks_filtering,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Finish Cooks filtering",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        return local_states, cooks_filtered_shared_state, round_idx
