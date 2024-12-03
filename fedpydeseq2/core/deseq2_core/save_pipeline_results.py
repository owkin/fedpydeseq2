"""Module to implement Mixin to get results as a shared state."""

import anndata as ad
from substrafl.remote import remote_data

from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote_data
from fedpydeseq2.core.utils.pass_on_results import AggPassOnResults


class SavePipelineResults(AggPassOnResults):
    """Mixin class to save pipeline results.

    Attributes
    ----------
    local_adata : AnnData
        Local AnnData object.

    results : dict
        Results to share.

    VARM_KEYS : list
        List of keys to extract from the varm attribute.

    UNS_KEYS : list
        List of keys to extract from the uns attribute.

    Methods
    -------
    save_pipeline_results
        Save the pipeline results.
        These results will be downloaded at the end of the pipeline.
        They are defined using the VARM_KEYS and UNS_KEYS attributes.

    get_results_from_local_states
        Get the results to share from the local states.

    """

    local_adata: ad.AnnData
    results: dict | None

    VARM_KEYS = [
        "MAP_dispersions",
        "dispersions",
        "genewise_dispersions",
        "non_zero",
        "fitted_dispersions",
        "LFC",
        "padj",
        "p_values",
        "wald_statistics",
        "wald_se",
        "replaced",
        "refitted",
    ]

    UNS_KEYS = [
        "prior_disp_var",
        "_squared_logres",
        "contrast",
    ]

    def save_pipeline_results(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        round_idx,
        clean_models,
    ):
        """Build the results that will be downloaded at the end of the pipeline.

        Parameters
        ----------
        train_data_nodes: list[TrainDataNode]
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
        local_states, shared_states, round_idx = local_step(
            local_method=self.get_results_from_local_states,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Get results to share from the local centers",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        # Build the global list of genes for which to replace outliers
        aggregation_step(
            aggregation_method=self.pass_on_results,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Merge the lists of results and return output",
            round_idx=round_idx,
            clean_models=False,
        )

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def get_results_from_local_states(
        self,
        data_from_opener,
        shared_state: dict | None,
    ) -> dict:
        """
        Get the results to share from the local states.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict, optional
            Not used.

        Returns
        -------
        dict
            Shared state containing the gene names, as well
            as selected fields from the varm and uns attributes.

        """
        shared_state = {
            "gene_names": self.local_adata.var_names,
        }
        for varm_key in self.VARM_KEYS:
            if varm_key in self.local_adata.varm.keys():
                shared_state[varm_key] = self.local_adata.varm[varm_key]
            else:
                shared_state[varm_key] = None

        for uns_key in self.UNS_KEYS:
            shared_state[uns_key] = self.local_adata.uns[uns_key]

        return shared_state
