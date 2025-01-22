"""Module containing the steps to compute trimmed mean."""

from typing import Literal

from fedpydeseq2.core.fed_algorithms.compute_trimmed_mean.substeps import (
    AggFinalTrimmedMean,
)
from fedpydeseq2.core.fed_algorithms.compute_trimmed_mean.substeps import (
    AggInitTrimmedMean,
)
from fedpydeseq2.core.fed_algorithms.compute_trimmed_mean.substeps import (
    AggIterationTrimmedMean,
)
from fedpydeseq2.core.fed_algorithms.compute_trimmed_mean.substeps import (
    LocalIterationTrimmedMean,
)
from fedpydeseq2.core.fed_algorithms.compute_trimmed_mean.substeps import (
    LocFinalTrimmedMean,
)
from fedpydeseq2.core.fed_algorithms.compute_trimmed_mean.substeps import (
    LocInitTrimmedMean,
)
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.logging.logging_decorators import end_iteration
from fedpydeseq2.core.utils.logging.logging_decorators import end_loop
from fedpydeseq2.core.utils.logging.logging_decorators import log_organisation_method
from fedpydeseq2.core.utils.logging.logging_decorators import start_iteration
from fedpydeseq2.core.utils.logging.logging_decorators import start_loop


class ComputeTrimmedMean(
    LocInitTrimmedMean,
    AggInitTrimmedMean,
    LocalIterationTrimmedMean,
    AggIterationTrimmedMean,
    LocFinalTrimmedMean,
    AggFinalTrimmedMean,
):
    """Strategy to compute the trimmed mean."""

    @log_organisation_method
    def compute_trim_mean(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        round_idx: int,
        clean_models: bool,
        layer_used: str,
        mode: Literal["normal", "cooks"] = "normal",
        trim_ratio: float | None = None,
        n_iter: int = 50,
        refit: bool = False,
        min_replicates_trimmed_mean: int = 3,
    ):
        """Run the trimmed mean computation on the layer specified.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        layer_used : str
            The layer used to compute the trimmed mean.

        local_states: dict
            Local states. Required to propagate intermediate results.

        round_idx: int
            The current round.

        clean_models: bool
            If True, the models are cleaned.

        mode : Literal["normal", "cooks"]
            The mode to use. If "cooks", the local trimmed mean is actually computed
            per level, and predefined trim ratios are applied, as well as certain
            scaling factors on the outputed means.
            If "normal", the local trimmed mean is computed on the whole dataset, as
            expected, using the trim_ratio parameter.

        trim_ratio : float, optional
            The ratio to trim. Should be between 0 and 0.5.
            Is only used in "normal" mode, and should be None in "cooks" mode.

        n_iter : int
            The number of iterations.

        refit : bool
            If True, the function will compute the trimmed mean on the refit adata only.

        min_replicates_trimmed_mean : int
            The minimum number of replicates to compute the trimmed mean.

        Returns
        -------
        local_states: list[dict]
            Local states dictionaries.

        final_trimmed_mean_agg_share_state: dict
            Dictionary containing the final trimmed mean aggregation share
            state in a field "trimmed_mean_<layer_used>".

        round_idx: int
        """
        if mode == "cooks":
            assert trim_ratio is None, "trim_ratio should be None in cooks mode"

        local_states, shared_states, round_idx = local_step(
            local_method=self.loc_init_trimmed_mean,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Initialize trim mean",
            round_idx=round_idx,
            clean_models=clean_models,
            method_params={
                "layer_used": layer_used,
                "mode": mode,
                "refit": refit,
                "min_replicates_trimmed_mean": min_replicates_trimmed_mean,
            },
        )

        aggregation_share_state, round_idx = aggregation_step(
            aggregation_method=self.agg_init_trimmed_mean,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Aggregation init of trimmed mean",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        start_loop()
        for iteration in range(n_iter):
            start_iteration(iteration)
            local_states, shared_states, round_idx = local_step(
                local_method=self.local_iteration_trimmed_mean,
                train_data_nodes=train_data_nodes,
                output_local_states=local_states,
                input_local_states=local_states,
                input_shared_state=aggregation_share_state,
                aggregation_id=aggregation_node.organization_id,
                description="Local iteration of trimmed mean",
                round_idx=round_idx,
                clean_models=clean_models,
                method_params={
                    "layer_used": layer_used,
                    "mode": mode,
                    "trim_ratio": trim_ratio,
                    "refit": refit,
                },
            )

            aggregation_share_state, round_idx = aggregation_step(
                aggregation_method=self.agg_iteration_trimmed_mean,
                train_data_nodes=train_data_nodes,
                aggregation_node=aggregation_node,
                input_shared_states=shared_states,
                description="Aggregation iteration of trimmed mean",
                round_idx=round_idx,
                clean_models=clean_models,
            )
            end_iteration()
        end_loop()

        local_states, shared_states, round_idx = local_step(
            local_method=self.final_local_trimmed_mean,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=aggregation_share_state,
            aggregation_id=aggregation_node.organization_id,
            description="Final local step of trimmed mean",
            round_idx=round_idx,
            clean_models=clean_models,
            method_params={
                "layer_used": layer_used,
                "trim_ratio": trim_ratio,
                "mode": mode,
                "refit": refit,
            },
        )

        final_trimmed_mean_agg_share_state, round_idx = aggregation_step(
            aggregation_method=self.final_agg_trimmed_mean,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="final aggregation of trimmed mean",
            round_idx=round_idx,
            clean_models=clean_models,
            method_params={"layer_used": layer_used, "mode": mode},
        )

        return local_states, final_trimmed_mean_agg_share_state, round_idx
