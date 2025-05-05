import anndata as ad
from substrafl.remote import remote_data

from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote_data
from fedpydeseq2.core.utils.logging.logging_decorators import log_organisation_method


class ReplaceRefittedValues:
    """Mixin class to replace refitted values."""

    local_adata: ad.AnnData | None
    refit_adata: ad.AnnData | None

    @log_organisation_method
    def replace_refitted_values(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        round_idx,
        clean_models,
    ):
        """Replace the values that were refitted in `local_adata`s.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: list[Dict]
            Local states. Required to propagate intermediate results.

        round_idx: int
            Index of the current round.

        clean_models: bool
            Whether to clean the models after the computation.

        Returns
        -------
        local_states: dict
            Local states, with refitted values

        round_idx: int
            The updated round index.
        """
        local_states, shared_states, round_idx = local_step(
            local_method=self.loc_replace_refitted_values,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Replace refitted values in local adatas",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        return local_states, round_idx

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def loc_replace_refitted_values(self, data_from_opener, shared_state):
        """Replace refitted values in local_adata from refit_adata.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict
            Not used.
        """
        # Replace values in main object
        list_varm_keys = [
            "_normed_means",
            "LFC",
            "genewise_dispersions",
            "fitted_dispersions",
            "MAP_dispersions",
            "dispersions",
        ]
        for key in list_varm_keys:
            self.local_adata.varm[key][self.local_adata.varm["refitted"]] = (
                self.refit_adata.varm[key]
            )

        # Take into account new all-zero genes
        new_all_zeroes_genes = self.local_adata.uns["new_all_zeroes_genes"]
        if len(new_all_zeroes_genes) > 0:
            self.local_adata.varm["_normed_means"][
                self.local_adata.var_names.get_indexer(new_all_zeroes_genes)
            ] = 0
            self.local_adata.varm["LFC"].loc[new_all_zeroes_genes, :] = 0
