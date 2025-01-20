"""Main module to compute dispersions by minimizing the MLE using a grid search."""

from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_MAP_dispersions.substeps import (  # noqa: E501
    LocFilterMAPDispersions,
)
from fedpydeseq2.core.fed_algorithms import ComputeDispersionsGridSearch
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.logging.logging_decorators import log_organisation_method


class ComputeMAPDispersions(
    LocFilterMAPDispersions,
    ComputeDispersionsGridSearch,
):
    """
    Mixin class to implement the computation of MAP dispersions.

    Methods
    -------
    fit_MAP_dispersions
        A method to fit the MAP dispersions and filter them.
        The filtering is done by removing the dispersions that are too far from the
        trend curve.

    """

    @log_organisation_method
    def fit_MAP_dispersions(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        shared_state,
        round_idx,
        clean_models,
        refit_mode: bool = False,
    ):
        """Fit MAP dispersions, and apply filtering.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: dict
            Local states. Required to propagate intermediate results.

        shared_state: dict
            Contains the output of the trend fitting,
            that is a dictionary with a "fitted_dispersion" field containing
            the fitted dispersions from the trend curve, a "prior_disp_var" field
            containing the prior variance of the dispersions, and a "_squared_logres"
            field containing the squared residuals of the trend fitting.

        round_idx: int
            The current round.

        clean_models: bool
            Whether to clean the models after the computation.

        refit_mode: bool
            Whether to run the pipeline in refit mode, after cooks outliers were
            replaced. If True, the pipeline will be run on `refit_adata`s instead of
            `local_adata`s. (default: False).


        Returns
        -------
        local_states: dict
            Local states. Required to propagate intermediate results.

        round_idx: int
            The updated round index.
        """
        local_states, shared_state, round_idx = self.fit_dispersions(
            train_data_nodes,
            aggregation_node,
            local_states,
            shared_state=shared_state,
            round_idx=round_idx,
            clean_models=clean_models,
            fit_mode="MAP",
            refit_mode=refit_mode,
        )

        # Filter the MAP dispersions.
        local_states, _, round_idx = local_step(
            local_method=self.filter_outlier_genes,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Filter MAP dispersions.",
            round_idx=round_idx,
            clean_models=clean_models,
            method_params={"refit_mode": refit_mode},
        )

        return local_states, round_idx
