import pickle as pkl
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pytest
from fedpydeseq2_datasets.utils import get_experiment_id
from fedpydeseq2_datasets.utils import get_ground_truth_dds_name
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.remote import remote_data

from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_MAP_dispersions.substeps import (  # noqa: E501
    LocFilterMAPDispersions,
)
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote_data
from fedpydeseq2.core.utils.pass_on_results import AggPassOnResults
from tests.tcga_testing_pipe import run_tcga_testing_pipe
from tests.unit_tests.unit_test_helpers.unit_tester import UnitTester


@pytest.mark.usefixtures(
    "raw_data_path", "local_processed_data_path", "tcga_assets_directory"
)
def test_MAP_dispersions_filtering_on_small_genes_small_samples(
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
):
    MAP_dispersions_filtering_testing_pipe(
        raw_data_path,
        local_processed_data_path,
        tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        small_samples=True,
        small_genes=True,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
        design_factors="stage",
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
    )


@pytest.mark.self_hosted_fast
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
def test_MAP_dispersions_filtering_on_small_samples_on_self_hosted_fast(
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
):
    MAP_dispersions_filtering_testing_pipe(
        raw_data_path,
        tmp_processed_data_path,
        tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        small_samples=True,
        small_genes=False,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
        design_factors="stage",
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
    )


@pytest.mark.self_hosted_slow
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
def test_MAP_dispersions_filtering_on_self_hosted_slow(
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
):
    MAP_dispersions_filtering_testing_pipe(
        raw_data_path,
        tmp_processed_data_path,
        tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        small_samples=False,
        small_genes=False,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
        design_factors="stage",
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
    )


def MAP_dispersions_filtering_testing_pipe(
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
    dataset_name="TCGA-LUAD",
    small_samples=True,
    small_genes=True,
    simulate=True,
    backend="subprocess",
    only_two_centers=False,
    design_factors: str | list[str] = "stage",
    ref_levels: dict[str, str] | None = {"stage": "Advanced"},  # noqa: B006
    reference_dds_ref_level: tuple[str, ...] | None = None,
):
    """Perform a unit test for MAP dispersions filtering.

    Starting with the same genewise and MAP dispersions as the reference dataset,
    filter dispersion outliers and compare the results with the reference.

    Parameters
    ----------
    raw_data_path: Path
        The path to the root data.

    local_processed_data_path: Path
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

    design_factors: str or list
        The design factors to use.

    ref_levels: dict or None
        The reference levels of the design factors.

    reference_dds_ref_level: tuple or None
        The reference level of the design factors.
    """

    # Setup the ground truth path.
    experiment_id = get_experiment_id(
        dataset_name,
        small_samples,
        small_genes,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=None,
    )

    reference_data_path = (
        local_processed_data_path / "centers_data" / "tcga" / experiment_id
    )
    # Get FL results.
    fl_results = run_tcga_testing_pipe(
        DispersionsFilteringTester(
            design_factors=design_factors,
            ref_levels=ref_levels,
            reference_data_path=reference_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
        ),
        raw_data_path=raw_data_path,
        processed_data_path=local_processed_data_path,
        assets_directory=tcga_assets_directory,
        simulate=simulate,
        dataset_name=dataset_name,
        small_samples=small_samples,
        small_genes=small_genes,
        backend=backend,
        only_two_centers=only_two_centers,
        register_data=True,
        design_factors=design_factors,
        reference_dds_ref_level=reference_dds_ref_level,
    )

    # pooled dds file name
    pooled_dds_file_name = get_ground_truth_dds_name(reference_dds_ref_level)

    pooled_dds_file_path = (
        local_processed_data_path
        / "pooled_data"
        / "tcga"
        / experiment_id
        / f"{pooled_dds_file_name}.pkl"
    )
    with open(pooled_dds_file_path, "rb") as file:
        pooled_dds = pkl.load(file)

    # Tests that the final dispersions are equal to the pooled ones
    assert np.allclose(
        pooled_dds.varm["dispersions"], fl_results["dispersions"], equal_nan=True
    )


class DispersionsFilteringTester(
    UnitTester,
    LocFilterMAPDispersions,
    AggPassOnResults,
):
    """A class to implement a unit test for MAP dispersions filtering.

    Parameters
    ----------
    design_factors : str or list
        Name of the columns of metadata to be used as design variables.
        If you are using categorical and continuous factors, you must put
        all of them here.

    ref_levels : dict or None
        An optional dictionary of the form ``{"factor": "test_level"}``
        specifying for each factor the reference (control) level against which
        we're testing, e.g. ``{"condition", "A"}``. Factors that are left out
        will be assigned random reference levels. (default: ``None``).

    continuous_factors : list or None
        An optional list of continuous (as opposed to categorical) factors. Any factor
        not in ``continuous_factors`` will be considered categorical
        (default: ``None``).

    num_jobs : int
        The number of jobs to use for local parallel processing in MLE tasks.
        (default: ``8``).

    reference_data_path : str or Path
        The path to the reference data. This is used to build the reference
        DeseqDataSet. This is only used for testing purposes, and should not be
        used in a real-world scenario.

    reference_dds_ref_level : tuple por None
        The reference level of the reference DeseqDataSet. This is used to build the
        reference DeseqDataSet. This is only used for testing purposes, and should not
        be used in a real-world scenario.

    Methods
    -------
    build_compute_plan
        Build the computation graph to perform and save MAP filtering.

    init_local_states
        Remote_data method.
        Copy the reference dds to the local state and add the non_zero mask.
        It returns a shared state with the MAP dispersions.

    get_filtered_dispersions
        Remote_data method.
        Get the filtered dispersions from the local adata.

    save_filtered_dispersions
        Save the filtered dispersions.
    """

    def __init__(
        self,
        design_factors: str | list[str],
        ref_levels: dict[str, str] | None = None,
        continuous_factors: list[str] | None = None,
        num_jobs=8,
        reference_data_path: str | Path | None = None,
        reference_dds_ref_level: tuple[str, ...] | None = None,
        *args,
        **kwargs,
    ):
        # Add all arguments to super init so that they can be retrieved by nodes.
        super().__init__(
            design_factors=design_factors,
            ref_levels=ref_levels,
            continuous_factors=continuous_factors,
            num_jobs=num_jobs,
            reference_data_path=reference_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
        )

        #### Define hyper parameters ####
        self.num_jobs = num_jobs

    def build_compute_plan(
        self,
        train_data_nodes: list[TrainDataNode],
        aggregation_node: AggregationNode,
        evaluation_strategy=None,
        num_rounds=None,
        clean_models=True,
    ):
        """Build the computation graph to perform and save MAP filtering.

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

        #### Create a reference DeseqDataset object ####

        local_states, shared_states, round_idx = self.set_local_reference_dataset(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=clean_models,
        )

        #### Load reference dataset as local_adata and set local states ####

        local_states, shared_states, round_idx = local_step(
            local_method=self.init_local_states,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Initialize local states",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        #### Filter MAP dispersions ####

        local_states, _, round_idx = local_step(
            local_method=self.filter_outlier_genes,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=shared_states[0],
            aggregation_id=aggregation_node.organization_id,
            description="Filter MAP dispersions.",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        self.save_filtered_dispersions(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=clean_models,
        )

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def init_local_states(
        self, data_from_opener: ad.AnnData, shared_state: Any
    ) -> dict:
        """Copy the reference dds to the local state and add the non_zero mask.

        It returns a shared state with the MAP dispersions.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : Any
            Shared state with "fitted_dispersions" and "prior_disp_var" keys.
        """

        self.local_adata = self.local_reference_dds.copy()
        # Delete the unused "_mu_hat" layer as it will raise errors
        del self.local_adata.layers["_mu_hat"]
        self.local_adata.varm["non_zero"] = self.local_reference_dds.varm["non_zero"]

        return {
            "MAP_dispersions": self.local_adata.varm["MAP_dispersions"],
        }

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def get_filtered_dispersions(self, data_from_opener, shared_state):
        """Get the filtered dispersions.

        Returns
        -------
        dict
            A dictionary with the filtered dispersions.
        """
        return {"dispersions": self.local_adata.varm["dispersions"]}

    def save_filtered_dispersions(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        round_idx,
        clean_models,
    ):
        """Save the filtered dispersions.

        This method gets the filtered dispersions from one of the local states and saves
        them in the results format.

        Parameters
        ----------
        train_data_nodes : list[TrainDataNode]
            List of train data nodes.

        aggregation_node : AggregationNode
            Aggregation node.

        local_states : dict
            Dictionary of local states.

        round_idx : int
            Round index.

        clean_models : bool
            Whether to clean the models after the computation.
            Note that the last step is not cleaned.
        """
        # ---- Get the filtered dispersions ---- #
        local_states, shared_states, round_idx = local_step(
            local_method=self.get_filtered_dispersions,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Get filtered dispersions.",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        # ---- Save the filtered dispersions ---- #

        aggregation_step(
            aggregation_method=self.pass_on_results,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Save filtered dispersions.",
            round_idx=round_idx,
            clean_models=False,
        )

        return local_states, round_idx
