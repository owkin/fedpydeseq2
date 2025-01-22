"""Main module to compute method of moments (MoM) dispersions."""

from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions.compute_MoM_dispersions.compute_rough_dispersions import (  # noqa: E501
    ComputeRoughDispersions,
)
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions.compute_MoM_dispersions.substeps import (  # noqa: E501
    AggMomentsDispersion,
)
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions.compute_MoM_dispersions.substeps import (  # noqa: E501
    LocInvSizeMean,
)
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.logging.logging_decorators import log_organisation_method


class ComputeMoMDispersions(
    ComputeRoughDispersions,
    LocInvSizeMean,
    AggMomentsDispersion,
):
    """Mixin class to implement the computation of MoM dispersions.

    Relies on the ComputeRoughDispersions class, in addition to substeps.

    Methods
    -------
    compute_MoM_dispersions
        The method to compute the MoM dispersions, that must be used in the main
        pipeline.
    """

    @log_organisation_method
    def compute_MoM_dispersions(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        gram_features_shared_states,
        round_idx,
        clean_models,
        refit_mode: bool = False,
    ):
        """Compute method of moments dispersions.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: dict
            Local states. Required to propagate intermediate results.

        gram_features_shared_states: list
            The list of shared states outputed by the compute_size_factors step.
            They contain a "local_gram_matrix" and a "local_features" fields.

        round_idx: int
            The current round.

        clean_models: bool
            Whether to clean the models after the computation.

        refit_mode: bool
            Whether to run the pipeline in refit mode, after cooks outliers were
            replaced. If True, the pipeline will be run on `refit_adata`s instead of
            `local_adata`s (default: False).

        Returns
        -------
        local_states: dict
            Local states. Required to propagate intermediate results.

        mom_dispersions_shared_state: dict
            Shared states containing MoM dispersions.

        round_idx: int
            The updated round number.
        """
        ###### Fit rough dispersions ######

        local_states, shared_states, round_idx = self.compute_rough_dispersions(
            train_data_nodes,
            aggregation_node,
            local_states,
            gram_features_shared_states=gram_features_shared_states,
            round_idx=round_idx,
            clean_models=clean_models,
            refit_mode=refit_mode,
        )

        ###### Compute moments dispersions ######

        # ---- Compute local means for moments dispersions---- #

        local_states, shared_states, round_idx = local_step(
            local_method=self.local_inverse_size_mean,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            round_idx=round_idx,
            input_local_states=local_states,
            input_shared_state=shared_states,
            aggregation_id=aggregation_node.organization_id,
            description="Compute local inverse size factor means.",
            clean_models=clean_models,
            method_params={"refit_mode": refit_mode},
        )

        # ---- Compute moments dispersions and merge to get MoM dispersions ---- #

        mom_dispersions_shared_state, round_idx = aggregation_step(
            aggregation_method=self.aggregate_moments_dispersions,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            round_idx=round_idx,
            description="Compute global MoM dispersions",
            clean_models=clean_models,
        )

        return local_states, mom_dispersions_shared_state, round_idx
