"""Module to test the correct saving of the results.

This module tests the fact that we recover the desired results both
in the simulation mode and in the subprocess mode, as they have
quite different behaviors.
"""

import pickle as pkl
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pytest
from substrafl import ComputePlanBuilder
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.deseq2_core.save_pipeline_results import SavePipelineResults
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data
from fedpydeseq2.core.utils.logging import log_save_local_state
from tests.tcga_testing_pipe import run_tcga_testing_pipe


@pytest.mark.usefixtures(
    "raw_data_path", "local_processed_data_path", "tcga_assets_directory"
)
def test_save_pipeline_results_on_small_genes(
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
):
    save_pipeline_results_testing_pipe(
        raw_data_path,
        processed_data_path=local_processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        small_samples=False,
        small_genes=True,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
    )


@pytest.mark.usefixtures(
    "raw_data_path", "local_processed_data_path", "tcga_assets_directory"
)
def test_save_pipeline_results_on_subprocess(
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
):
    save_pipeline_results_testing_pipe(
        raw_data_path,
        processed_data_path=local_processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        small_samples=True,
        small_genes=True,
        simulate=False,
        backend="subprocess",
        only_two_centers=False,
    )


@pytest.mark.self_hosted_fast
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
def test_save_pipeline_results_on_small_samples_on_self_hosted_fast(
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
):
    save_pipeline_results_testing_pipe(
        raw_data_path,
        processed_data_path=tmp_processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        small_samples=True,
        small_genes=False,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
    )


@pytest.mark.self_hosted_slow
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
def test_save_pipeline_results_on_self_hosted_slow(
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
):
    save_pipeline_results_testing_pipe(
        raw_data_path,
        processed_data_path=tmp_processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        small_samples=False,
        small_genes=False,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
    )


@pytest.mark.self_hosted_slow
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
def test_save_pipeline_results_on_subprocess_on_self_hosted_slow(
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
):
    save_pipeline_results_testing_pipe(
        raw_data_path,
        processed_data_path=tmp_processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        small_samples=True,
        small_genes=True,
        simulate=False,
        backend="subprocess",
        only_two_centers=False,
    )


def save_pipeline_results_testing_pipe(
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
    dataset_name="TCGA-LUAD",
    small_samples=True,
    small_genes=True,
    simulate=True,
    backend="subprocess",
    only_two_centers=False,
):
    """Test the SavePipelineResults class.



    Parameters
    ----------
    raw_data_path: Path
        The path to the root data.

    processed_data_path: Path
        The path to the processed data. The subdirectories will
        be created if needed

    tcga_assets_directory: Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.

    dataset_name: TCGADatasetNames
        The name of the dataset, for example "TCGA-LUAD".

    small_samples: bool
        Whether to use a small number of samples.
        If True, the number of samples is reduced to 10 per center.

    small_genes: bool
        Whether to use a small number of genes.
        If True, the number of genes is reduced to 100.

    simulate: bool
        If true, use the simulation mode, otherwise use the subprocess mode.

    backend: str
        The backend to use. Either "subprocess" or "docker".

    only_two_centers: bool
        If true, restrict the data to two centers.

    """
    fl_results = run_tcga_testing_pipe(
        SavePipelineResultsTester(),
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path,
        assets_directory=tcga_assets_directory,
        simulate=simulate,
        dataset_name=dataset_name,
        small_samples=small_samples,
        small_genes=small_genes,
        backend=backend,
        only_two_centers=only_two_centers,
        register_data=True,
        reference_dds_ref_level=None,
    )

    # Check that all fields are present
    assert isinstance(fl_results, dict)
    for key in SavePipelineResults.VARM_KEYS:
        assert key in fl_results
    for key in SavePipelineResults.UNS_KEYS:
        assert key in fl_results
    assert "gene_names" in fl_results


class SavePipelineResultsTester(ComputePlanBuilder, SavePipelineResults):
    """Tester for the SavePipelineResults class.

    Parameters
    ----------
    n_rounds : int
        Number of rounds.

    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # Add all arguments to super init so that they can be retrieved by nodes.
        super().__init__()

        self.local_adata: ad.AnnData | None = None
        self.results: dict | None = None
        self.refit_adata: ad.AnnData | None = None

    def build_compute_plan(
        self,
        train_data_nodes: list[TrainDataNode],
        aggregation_node: AggregationNode,
        evaluation_strategy=None,
        num_rounds=None,
        clean_models=True,
    ):
        """Build the computation graph to run a DESeq2 pipe.

        Parameters
        ----------
        train_data_nodes : List[TrainDataNode]
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

        #### Load the opener data local_adata and set local states ####

        local_states, shared_states, round_idx = local_step(
            local_method=self.init_local_states,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=None,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Initialize local states",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        shared_state, round_idx = aggregation_step(
            aggregation_method=self.create_all_fields_with_dummies,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Create all fields with dummies",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        local_states, shared_states, round_idx = local_step(
            local_method=self.set_dummies_in_local_adata,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Set dummies in local adata",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        self.save_pipeline_results(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            round_idx=round_idx,
            clean_models=clean_models,
        )

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def init_local_states(
        self, data_from_opener: ad.AnnData, shared_state: Any
    ) -> dict:
        """Copy the reference dds and add the total number of samples.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : Any
            Shared state with a "num_samples" key.
        """

        self.local_adata = data_from_opener.copy()
        return {"num_vars": self.local_adata.n_vars}

    @remote
    @log_remote
    def create_all_fields_with_dummies(self, shared_states: dict):
        """Set all fields with dummies. Used for testing only."""
        num_vars = shared_states[0]["num_vars"]
        varm_dummies = {}

        if "replaced" in self.VARM_KEYS:
            varm_dummies["replaced"] = np.random.rand(num_vars) > 0.5
        if "refitted" in self.VARM_KEYS:
            assert "replaced" in self.VARM_KEYS
            varm_dummies["replaced"] = np.random.rand(num_vars) > 0.5
            varm_dummies["refitted"] = varm_dummies["replaced"] & (
                np.random.rand(num_vars) > 0.5
            )

        for varm_key in self.VARM_KEYS:
            if varm_key in {"refitted", "replaced"}:
                # Create a random boolean array of size num_vars
                continue
            varm_dummies[varm_key] = np.random.rand(num_vars, 2)

        uns_dummmies = {}
        for uns_key in self.UNS_KEYS:
            uns_dummmies[uns_key] = np.random.rand(2)

        return {"varm_dummies": varm_dummies, "uns_dummmies": uns_dummmies}

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def set_dummies_in_local_adata(
        self,
        data_from_opener,
        shared_state: dict,
    ) -> dict:
        """Create a dummy array.

        Parameters
        ----------
        data_from_opener : Any
            Not used.

        shared_state : dict
            Shared state with "varm_dummies" and "uns_dummmies" keys.

        """
        for varm_key, varm_dummy in shared_state["varm_dummies"].items():
            self.local_adata.varm[varm_key] = varm_dummy

        for uns_key, uns_dummy in shared_state["uns_dummmies"].items():
            self.local_adata.uns[uns_key] = uns_dummy

        print("Self refit adata")
        print(self.refit_adata)

        return {"num_vars": self.local_adata.n_vars}

    @log_save_local_state
    def save_local_state(self, path: Path) -> None:
        """Save the local state of the strategy.

        Parameters
        ----------
        path : Path
            Path to the file where to save the state. Automatically handled by subtrafl.
        """
        state_to_save = {
            "results": self.results,
            "local_adata": self.local_adata,
            "refit_adata": self.refit_adata,
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

        self.results = state_to_load["results"]
        self.local_adata = state_to_load["local_adata"]
        self.refit_adata = state_to_load["refit_adata"]

        return self

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
