from fedpydeseq2.core.deseq2_core.replace_outliers.substeps import AggMergeOutlierGenes
from fedpydeseq2.core.deseq2_core.replace_outliers.substeps import AggNewAllZeros
from fedpydeseq2.core.deseq2_core.replace_outliers.substeps import LocFindCooksOutliers
from fedpydeseq2.core.deseq2_core.replace_outliers.substeps import (
    LocReplaceCooksOutliers,
)
from fedpydeseq2.core.deseq2_core.replace_outliers.substeps import (
    LocSetNewAllZerosAndGetFeatures,
)
from fedpydeseq2.core.deseq2_core.replace_outliers.substeps import LocSetRefitAdata
from fedpydeseq2.core.fed_algorithms import ComputeTrimmedMean
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.logging.logging_decorators import log_organisation_method


class ReplaceCooksOutliers(
    ComputeTrimmedMean,
    LocFindCooksOutliers,
    AggMergeOutlierGenes,
    LocReplaceCooksOutliers,
    LocSetRefitAdata,
    AggNewAllZeros,
    LocSetNewAllZerosAndGetFeatures,
):
    """Mixin class to replace Cook's outliers."""

    trimmed_mean_num_iter: int

    @log_organisation_method
    def replace_outliers(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        cooks_shared_state,
        round_idx,
        clean_models,
    ):
        """Replace outlier counts.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: list[dict]
            Local states. Required to propagate intermediate results.

        cooks_shared_state: dict
            Shared state with the dispersion values for Cook's distances, in a
            "cooks_dispersions" key.


        round_idx: int
            Index of the current round.

        clean_models: bool
            Whether to clean the models after the computation.

        Returns
        -------
        local_states: dict
            Local states. The new local state contains Cook's distances.

        shared_states: list[dict]
            List of shared states with the features vector to input to
            compute_genewise_dispersion in a "local_features" key.

        round_idx: int
            The updated round index.
        """
        # Store trimmed means and find local Cooks outliers
        local_states, shared_states, round_idx = local_step(
            local_method=self.loc_find_cooks_outliers,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=cooks_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Find local Cooks outliers",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        # Build the global list of genes for which to replace outliers
        genes_to_replace_share_state, round_idx = aggregation_step(
            aggregation_method=self.agg_merge_outlier_genes,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Merge the lists of local outlier genes",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        # Store trimmed means and find local Cooks outliers
        local_states, shared_states, round_idx = local_step(
            local_method=self.loc_set_refit_adata,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=genes_to_replace_share_state,
            aggregation_id=aggregation_node.organization_id,
            description="Set the refit adata with the genes to replace",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        # Compute imputation values, on genes to refit only.
        local_states, trimmed_means_shared_state, round_idx = self.compute_trim_mean(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=clean_models,
            layer_used="normed_counts",
            trim_ratio=0.2,
            mode="normal",
            n_iter=self.trimmed_mean_num_iter,
            refit=True,
        )

        # Replace outliers in replaceable samples locally
        local_states, shared_states, round_idx = local_step(
            local_method=self.loc_replace_cooks_outliers,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=trimmed_means_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Replace Cooks outliers locally",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        # Find genes who have only have zero counts due to imputation

        new_all_zeros_shared_state, round_idx = aggregation_step(
            aggregation_method=self.aggregate_new_all_zeros,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Find new all zero genes",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        # Set new all zeros genes and get features vector

        local_states, shared_states, round_idx = local_step(
            local_method=self.local_set_new_all_zeros_get_features,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=new_all_zeros_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Set new all zero genes and get features vector",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        return local_states, shared_states, round_idx
