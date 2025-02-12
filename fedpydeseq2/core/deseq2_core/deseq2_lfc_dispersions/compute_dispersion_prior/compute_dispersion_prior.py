"""Module containing the steps for fitting the dispersion trend."""
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_dispersion_prior.substeps import (  # noqa: E501
    AggFitDispersionTrendAndPrior,
)
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_dispersion_prior.substeps import (  # noqa: E501
    LocGetMeanDispersionAndMean,
)
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_dispersion_prior.substeps import (  # noqa: E501
    LocUpdateFittedDispersions,
)
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.logging.logging_decorators import log_organisation_method


class ComputeDispersionPrior(
    AggFitDispersionTrendAndPrior,
    LocGetMeanDispersionAndMean,
    LocUpdateFittedDispersions,
):
    """Mixin class to implement the fit of the dispersion trend.

    Methods
    -------
    compute_dispersion_prior
        The method to fit the dispersion trend.
    """

    @log_organisation_method
    def compute_dispersion_prior(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        genewise_dispersions_shared_state,
        round_idx,
        clean_models,
    ):
        """Fit the dispersion trend.

        Parameters
        ----------
        train_data_nodes: list
            list of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: dict
            Local states. Required to propagate intermediate results.

        genewise_dispersions_shared_state: dict
            Shared state with a "genewise_dispersions" key.

        round_idx: int
            Index of the current round.

        clean_models: bool
            Whether to clean the models after the computation.

        Returns
        -------
        local_states: dict
            Local states. Required to propagate intermediate results.

        dispersion_trend_share_state: dict
            Shared states with:
            - "fitted_dispersions": the fitted dispersions,
            - "prior_disp_var": the prior dispersion variance.

        round_idx: int
            The updated round index.
        """
        # --- Return means and dispersions ---#
        local_states, shared_states, round_idx = local_step(
            local_method=self.get_local_mean_and_dispersion,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            round_idx=round_idx,
            input_local_states=local_states,
            input_shared_state=genewise_dispersions_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Get local means and dispersions",
            clean_models=clean_models,
        )

        # ---- Fit dispersion trend ----#

        dispersion_trend_shared_state, round_idx = aggregation_step(
            aggregation_method=self.agg_fit_dispersion_trend_and_prior_dispersion,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            round_idx=round_idx,
            description="Fitting dispersion trend",
            clean_models=clean_models,
        )

        return local_states, dispersion_trend_shared_state, round_idx
