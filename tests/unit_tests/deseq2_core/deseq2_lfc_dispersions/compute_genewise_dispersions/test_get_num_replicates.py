"""Module to test the GetNumReplicates class."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fedpydeseq2_datasets.utils import get_experiment_id
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions.get_num_replicates import (  # noqa: E501
    GetNumReplicates,
)
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data
from tests.tcga_testing_pipe import run_tcga_testing_pipe
from tests.unit_tests.unit_test_helpers.levels import make_reference_and_fl_ref_levels
from tests.unit_tests.unit_test_helpers.unit_tester import UnitTester


@pytest.mark.parametrize(
    "design_factors, continuous_factors",
    [
        ("stage", None),
        (["stage", "gender"], None),
        (["stage", "gender", "CPE"], ["CPE"]),
    ],
)
@pytest.mark.usefixtures(
    "raw_data_path", "local_processed_data_path", "tcga_assets_directory"
)
def test_get_num_replicates_on_small_genes_small_samples(
    design_factors,
    continuous_factors,
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
):
    """
    Test the GetNumReplicates class on a small number of genes and samples.

    Parameters
    ----------
    design_factors : str or list
        The design factors to use.

    continuous_factors : list or None
        The continuous factors to use.

    raw_data_path : Path
        The path to the raw data.

    local_processed_data_path : Path
        The path to the processed data.

    tcga_assets_directory : Path
        The path to the assets directory.

    """
    get_num_replicates_testing_pipe(
        raw_data_path,
        local_processed_data_path,
        tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        small_samples=True,
        small_genes=True,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
    )


@pytest.mark.parametrize(
    "design_factors, continuous_factors",
    [
        ("stage", None),
        (["stage", "gender", "CPE"], ["CPE"]),
    ],
)
@pytest.mark.self_hosted_fast
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
def test_get_num_replicates_on_small_genes_on_self_hosted_fast(
    design_factors,
    continuous_factors,
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
):
    """
    Test the GetNumReplicates class on a small number of genes.

    Parameters
    ----------
    design_factors : str or list
        The design factors to use.

    continuous_factors : list or None
        The continuous factors to use.

    raw_data_path : Path
        The path to the raw data.

    tmp_processed_data_path : Path
        The path to the processed data.

    tcga_assets_directory : Path
        The path to the assets directory.

    """
    get_num_replicates_testing_pipe(
        raw_data_path,
        tmp_processed_data_path,
        tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        small_samples=False,
        small_genes=True,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
    )


def get_num_replicates_testing_pipe(
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
    continuous_factors: list[str] | None = None,
    ref_levels: dict[str, str] | None = {"stage": "Advanced"},  # noqa: B006
    reference_dds_ref_level: tuple[str, ...] | None = None,
):
    """Check that the GetNumReplicates class is working correctly.

    More specifically, we check that the num_replicates field indeed corresponds
    to the pandas value_counts of the full design matrix.

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

    continuous_factors: list or None
        The continuous factors to use.

    ref_levels: dict or None
        The reference levels of the design factors.

    reference_dds_ref_level: tuple or None
        The reference level of the design factors.
    """
    complete_ref_levels, reference_dds_ref_level = make_reference_and_fl_ref_levels(
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        ref_levels=ref_levels,
        reference_dds_ref_level=reference_dds_ref_level,
    )

    # Setup the ground truth path.
    experiment_id = get_experiment_id(
        dataset_name,
        small_samples,
        small_genes,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
    )

    reference_data_path = (
        local_processed_data_path / "centers_data" / "tcga" / experiment_id
    )
    # The test happens inside the last aggregation
    run_tcga_testing_pipe(
        GetNumReplicatesTester(
            design_factors=design_factors,
            continuous_factors=continuous_factors,
            ref_levels=complete_ref_levels,
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
        continuous_factors=continuous_factors,
        reference_dds_ref_level=reference_dds_ref_level,
    )


class GetNumReplicatesTester(UnitTester, GetNumReplicates):
    """A class to implement a unit test for the GetNumReplicates class.

    This class checks that that num_replicates indeed corresponds to the
    pandas value_counts of the full design matrix.

    To do so, we associate the num_replicate field to the corresponding
    design matrix lines, before checking that the series obtained thus
    matches the value_counts of the full design matrix.

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
        Build the computation graph to check the get_num_replicates function.

    get_local_cells_design
        Get the local cells and design.

    agg_check_value_counts
        Aggregate the value counts from the cells, design and num_replicates.
        This function performs the following steps:
        - Create the value_counts that result from the aggregation of the cells, design
            and num_replicates, i.e., the value counts seen by the local data (in the
            fl_count columns)
        - Create the full design matrix and compute the real value_counts from the full
            design matrix.
        - Check that the value counts are the same, that is that get_num_replicates is
            working correctly.

    """

    def __init__(
        self,
        design_factors: str | list[str],
        ref_levels: dict[str, str] | None = None,
        continuous_factors: list[str] | None = None,
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
            reference_data_path=reference_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
        )

        #### Define hyper parameters ####

    def build_compute_plan(
        self,
        train_data_nodes: list[TrainDataNode],
        aggregation_node: AggregationNode,
        evaluation_strategy=None,
        num_rounds=None,
        clean_models=True,
    ):
        """Build the computation graph to check the get_num_replicates function.

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

        (
            local_states,
            round_idx,
        ) = self.get_num_replicates(
            train_data_nodes, aggregation_node, local_states, round_idx, clean_models
        )

        local_states, shared_states, round_idx = local_step(
            local_method=self.get_local_cells_design,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Get local cells and design",
            round_idx=round_idx,
            clean_models=clean_models,
        )
        aggregation_step(
            aggregation_method=self.agg_check_value_counts,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Check matching value counts",
            round_idx=round_idx,
            clean_models=False,
        )

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def get_local_cells_design(self, data_from_opener, shared_state: dict) -> dict:
        """Get the local cells and design.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict
            Not used

        Returns
        -------
        dict
            Dictionary with the following keys:
            - cells: cells in the local data
            - design: design matrix
            - num_replicates: number of replicates
        """
        cells = self.local_adata.obs["cells"]
        design = self.local_adata.obsm["design_matrix"]
        num_replicates = self.local_adata.uns["num_replicates"]

        return {"cells": cells, "design": design, "num_replicates": num_replicates}

    @remote
    @log_remote
    def agg_check_value_counts(self, shared_states: list[dict]):
        """Aggregate the value counts from the cells, design and num_replicates.

        This function creates the value_counts that result from the
        aggregation of the cells, design and num_replicates, i.e., the value
        counts seen by the local data (in the fl_count columns)

        It then creates the full design matrix and computes the real value_counts
        from the full design matrix.

        It then checks that the value counts are the same, that is that
        get_num_replicates is working correctly.

        Parameters
        ----------
        shared_states : list[dict]
            List of shared states. Must contain the following keys:
            - cells: cells in the local data
            - design: design matrix
            - num_replicates: number of replicates

        """
        design = pd.concat(
            [shared_state["design"] for shared_state in shared_states], axis=0
        )
        cells = pd.concat(
            [shared_state["cells"] for shared_state in shared_states], axis=0
        )
        num_replicates = shared_states[0]["num_replicates"]
        design_columns = design.columns.tolist()
        counts_col = pd.Series(
            num_replicates.loc[cells].values, name="fl_count", index=cells.index
        )
        extended_design = pd.concat([design, counts_col], axis=1)
        # drop duplicates
        extended_design = extended_design.drop_duplicates().reset_index(drop=True)
        # On the other hand, compute value counts
        value_counts = design.value_counts().reset_index()
        # merge the two datasets
        merged = pd.merge(value_counts, extended_design, on=design_columns, how="left")
        # assert that the counts are the same
        assert np.allclose(merged["fl_count"].values, merged["count"].values)
