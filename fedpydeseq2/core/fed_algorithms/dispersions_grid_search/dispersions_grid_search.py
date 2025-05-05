"""Main module to compute dispersions by minimizing the MLE using a grid search."""

from typing import Literal

from fedpydeseq2.core.fed_algorithms.dispersions_grid_search.substeps import (
    AggGridUpdate,
)
from fedpydeseq2.core.fed_algorithms.dispersions_grid_search.substeps import LocGridLoss
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.logging.logging_decorators import end_iteration
from fedpydeseq2.core.utils.logging.logging_decorators import end_loop
from fedpydeseq2.core.utils.logging.logging_decorators import log_organisation_method
from fedpydeseq2.core.utils.logging.logging_decorators import start_iteration
from fedpydeseq2.core.utils.logging.logging_decorators import start_loop


class ComputeDispersionsGridSearch(
    AggGridUpdate,
    LocGridLoss,
):
    """Mixin class to implement the computation of genewise dispersions.

    The switch between genewise and MAP dispersions is done by setting the `fit_mode`
    argument in the `fit_dispersions` to either "MLE" or "MAP".

    Methods
    -------
    fit_dispersions
        A method to fit dispersions using grid search.
    """

    grid_batch_size: int
    grid_depth: int
    grid_length: int

    @log_organisation_method
    def fit_dispersions(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        shared_state,
        round_idx,
        clean_models,
        fit_mode: Literal["MLE", "MAP"] = "MLE",
        refit_mode: bool = False,
    ):
        """Fit dispersions using grid search.

        Supports two modes: "MLE", to fit gene-wise dispersions, and "MAP", to fit
        MAP dispersions and filter them to avoid shrinking the dispersions of genes
        that are too far from the trend curve.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: dict
            Local states. Required to propagate intermediate results.

        shared_state: dict, optional
            If the fit_mode is "MLE", it is None.
            If the fit_mode is "MAP", it contains the output of the trend fitting,
            that is a dictionary with a "fitted_dispersion" field containing
            the fitted dispersions from the trend curve, a "prior_disp_var" field
            containing the prior variance of the dispersions, and a "_squared_logres"
            field containing the squared residuals of the trend fitting.


        round_idx: int
            The current round.

        clean_models: bool
            Whether to clean the models after the computation.

        fit_mode: str
            If "MLE", gene-wise dispersions are fitted independently, and
            `"genewise_dispersions"` fields are populated. If "MAP", prior
            regularization is applied, `"MAP_dispersions"` fields are populated.

        refit_mode: bool
            Whether to run on `refit_adata`s instead of `local_adata`s (default: False).

        Returns
        -------
        local_states: dict
            Local states. Required to propagate intermediate results.

        shared_state: dict or list[dict]
            A dictionary containing:
            - "genewise_dispersions": The MLE dispersions, to be stored locally at
            - "lower_log_bounds": log lower bounds for the grid search (only used in
            internal loop),
            - "upper_log_bounds": log upper bounds for the grid search (only used in
            internal loop).

        round_idx: int
            The updated round index.
        """
        start_loop()
        for iteration in range(self.grid_depth):
            start_iteration(iteration)
            # Compute local loss summands at all grid points.
            local_states, shared_states, round_idx = local_step(
                local_method=self.local_grid_loss,
                train_data_nodes=train_data_nodes,
                output_local_states=local_states,
                input_local_states=local_states,
                input_shared_state=shared_state,
                aggregation_id=aggregation_node.organization_id,
                description="Compute local grid loss summands.",
                round_idx=round_idx,
                clean_models=clean_models,
                method_params={
                    "prior_reg": fit_mode == "MAP",
                    "refit_mode": refit_mode,
                },
            )

            # Aggregate local summands and refine the search interval.
            shared_state, round_idx = aggregation_step(
                aggregation_method=self.global_grid_update,
                train_data_nodes=train_data_nodes,
                aggregation_node=aggregation_node,
                input_shared_states=shared_states,
                description="Perform a global grid search update.",
                round_idx=round_idx,
                clean_models=clean_models,
                method_params={
                    "prior_reg": fit_mode == "MAP",
                    "dispersions_param_name": (
                        "genewise_dispersions"
                        if fit_mode == "MLE"
                        else "MAP_dispersions"
                    ),
                },
            )
            end_iteration()
        end_loop()

        return local_states, shared_state, round_idx
