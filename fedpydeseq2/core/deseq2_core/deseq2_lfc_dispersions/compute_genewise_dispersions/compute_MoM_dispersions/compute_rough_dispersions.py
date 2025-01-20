"""Module to compute rough dispersions."""

from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions.compute_MoM_dispersions.substeps import (  # noqa: E501
    AggCreateRoughDispersionsSystem,
)
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions.compute_MoM_dispersions.substeps import (  # noqa: E501
    AggRoughDispersion,
)
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions.compute_MoM_dispersions.substeps import (  # noqa: E501
    LocRoughDispersion,
)
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.logging.logging_decorators import log_organisation_method


class ComputeRoughDispersions(
    AggRoughDispersion,
    LocRoughDispersion,
    AggCreateRoughDispersionsSystem,
):
    """Mixin class to implement the computation of rough dispersions.

    Methods
    -------
    compute_rough_dispersions
        The method to compute the rough dispersions, that must be used in the main
        pipeline.

    """

    @log_organisation_method
    def compute_rough_dispersions(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        gram_features_shared_states,
        round_idx,
        clean_models,
        refit_mode: bool = False,
    ):
        """Compute rough dispersions.

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

        rough_dispersion_shared_state: dict
            Shared states containing rough dispersions.

        round_idx: int
            The updated round number.

        """
        # TODO: in refit mode, we need to gather the gram matrix and the features some
        #  way

        # ---- Solve global linear system ---- #

        rough_dispersion_system_shared_state, round_idx = aggregation_step(
            aggregation_method=self.create_rough_dispersions_system,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=gram_features_shared_states,
            round_idx=round_idx,
            description="Solving system for rough dispersions",
            clean_models=clean_models,
            method_params={"refit_mode": refit_mode},
        )

        # ---- Compute local rough dispersions---- #

        local_states, shared_states, round_idx = local_step(
            local_method=self.local_rough_dispersions,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            round_idx=round_idx,
            input_local_states=local_states,
            input_shared_state=rough_dispersion_system_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Computing local rough dispersions",
            clean_models=clean_models,
            method_params={"refit_mode": refit_mode},
        )

        # ---- Compute global rough dispersions---- #

        rough_dispersion_shared_state, round_idx = aggregation_step(
            aggregation_method=self.aggregate_rough_dispersions,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            round_idx=round_idx,
            description="Compute global rough dispersions",
            clean_models=clean_models,
        )

        return local_states, rough_dispersion_shared_state, round_idx
