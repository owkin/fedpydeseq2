import pickle as pkl
from pathlib import Path
from typing import Any

import anndata as ad
from substrafl import ComputePlanBuilder
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.remote import remote_data

from fedpydeseq2.core.fed_algorithms import ComputeTrimmedMean
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote_data
from fedpydeseq2.core.utils.logging import log_save_local_state
from fedpydeseq2.core.utils.pass_on_results import AggPassOnResults
from tests.unit_tests.unit_test_helpers.set_local_reference import SetLocalReference


class TrimmedMeanStrategyForTesting(
    ComputePlanBuilder, ComputeTrimmedMean, SetLocalReference, AggPassOnResults
):
    def __init__(
        self,
        trim_ratio: float,
        layer_used: str,
        nb_iter: int,
        refit: bool = False,
        save_layers_to_disk: bool = False,
        *args,
        **kwargs,
    ):
        self.results = {}
        self.trim_ratio = trim_ratio
        self.layer_used = layer_used
        self.nb_iter = nb_iter
        self.refit = refit
        self.reference_data_path = None
        self.local_adata: ad.AnnData | None = None
        self.refit_adata: ad.AnnData | None = None
        self.results: dict | None = None
        super().__init__()

        #### Save layers to disk
        self.save_layers_to_disk = save_layers_to_disk

        self.layers_to_save_on_disk = {
            "local_adata": [layer_used],
            "refit_adata": [layer_used],
        }

    def build_compute_plan(
        self,
        train_data_nodes: list[TrainDataNode],
        aggregation_node: AggregationNode,
        evaluation_strategy=None,
        num_rounds=None,
        clean_models=True,
    ):
        """Build the computation graph to run a FedDESeq2 pipe.

        Parameters
        ----------
        train_data_nodes : list[TrainDataNode]
            List of the train nodes.
        aggregation_node : AggregationNode
            Aggregation node.
        evaluation_strategy : EvaluationStrategy
            Not used.
        num_rounds : int
            Number of rounds. Not used.
        clean_models : bool
            Whether to clean the models after the computation. (default: ``False``).
        """
        round_idx = 0
        local_states: dict[str, LocalStateRef] = {}

        local_states, shared_states, round_idx = local_step(
            local_method=self.init_local_states_from_opener,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=None,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Initialize local states",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        (
            local_states,
            final_trimmed_mean_agg_share_state,
            round_idx,
        ) = self.compute_trim_mean(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models,
            layer_used=self.layer_used,
            mode="normal",
            trim_ratio=self.trim_ratio,
            n_iter=self.nb_iter,
            refit=self.refit,
        )

        aggregation_step(
            aggregation_method=self.pass_on_results,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=[final_trimmed_mean_agg_share_state],
            round_idx=round_idx,
            clean_models=False,
        )

    @log_save_local_state
    def save_local_state(self, path: Path) -> None:
        """Save the local state of the strategy.

        Parameters
        ----------
        path : Path
            Path to the file where to save the state. Automatically handled by subtrafl.
        """
        state_to_save = {
            "local_adata": self.local_adata,
            "refit_adata": self.refit_adata,
            "results": self.results,
        }
        with open(path, "wb") as file:
            pkl.dump(state_to_save, file)

    def load_local_state(self, path: Path) -> Any:
        """Load the local state of the strategy.

        Parameters
        ----------
        path : Path
            Path to the file where to load the state from. Automatically handled by
            subtrafl.
        """
        with open(path, "rb") as file:
            state_to_load = pkl.load(file)

        self.local_adata = state_to_load["local_adata"]
        self.refit_adata = state_to_load["refit_adata"]
        self.results = state_to_load["results"]

        return self

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def init_local_states_from_opener(
        self, data_from_opener: ad.AnnData, shared_state: Any
    ):
        """Copy the reference dds to the local state.

        If necessary, to overwrite in child classes to add relevant local states.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.
        shared_state : Any
            Shared state. Not used.
        """

        self.local_adata = data_from_opener.copy()
        # Subsample the first 10 columns to use as fake outlier genes
        self.refit_adata = self.local_adata[:, :10].copy()

    @property
    def num_round(self):
        """Return the number of round in the strategy.

        TODO do something clever with this.

        Returns
        -------
        int
            Number of round in the strategy.
        """
        return None

    def get_result(self):
        """Return the statistic computed.

        Returns
        -------
        dict
            The global statistics.
        """
        return self.results
