import pickle as pkl
from pathlib import Path

import pandas as pd
import pytest
from fedpydeseq2_datasets.utils import get_experiment_id
from fedpydeseq2_datasets.utils import get_ground_truth_dds_name
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.deseq2_core.compute_cook_distance import ComputeCookDistances
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
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_cooks_distances_on_small_genes(
    design_factors,
    continuous_factors,
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
):
    """Test Cook's distances on a small number of genes.

    Note that the first subcase is particularly important, as it tests the case where
    the number of replicates for all levels of the design is greater than 3.
    """
    cooks_distances_testing_pipe(
        raw_data_path,
        processed_data_path=processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
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


@pytest.mark.parametrize(
    "design_factors, continuous_factors",
    [
        ("stage", None),
        (["stage", "gender"], None),
        (["stage", "gender", "CPE"], ["CPE"]),
    ],
)
@pytest.mark.self_hosted_fast
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_cooks_distances_on_small_samples_on_self_hosted_fast(
    design_factors,
    continuous_factors,
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
):
    """Test Cook's distances on a small number of samples on a self hosted runner.

    This test is quite important for the (["stage", "gender", "CPE"], ["CPE"]) subset,
    as it tests the case where the number of replicates is less than 3 for all levels
    of the design, and we enter in this specific case of the computation of the trimmed
    variance.
    """
    cooks_distances_testing_pipe(
        raw_data_path,
        processed_data_path=processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        small_samples=True,
        small_genes=False,
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
        (["stage", "gender"], None),
        (["stage", "gender", "CPE"], ["CPE"]),
    ],
)
@pytest.mark.self_hosted_slow
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_cooks_distances_on_self_hosted_slow(
    design_factors,
    continuous_factors,
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
):
    """Test Cook's distances on a self hosted runner.

    This test is particularly important for the (["stage", "gender", "CPE"], ["CPE"])
    subcase, as it tests the case where some levels of the design have less than 3
    replicates while other have more. This means that only part of the levels are taken
    into account into the computation of the trimmed variance.
    """
    cooks_distances_testing_pipe(
        raw_data_path,
        processed_data_path=processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        small_samples=False,
        small_genes=False,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
    )


def cooks_distances_testing_pipe(
    raw_data_path,
    processed_data_path,
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
    """Perform a unit test for Cook's distances.

    Starting with the same counts as the reference dataset, compute sCook's distances
    and compare the results with the reference.

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

    reference_data_path = processed_data_path / "centers_data" / "tcga" / experiment_id

    fl_results = run_tcga_testing_pipe(
        CooksDistanceTester(
            design_factors=design_factors,
            ref_levels=complete_ref_levels,
            continuous_factors=continuous_factors,
            reference_data_path=reference_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
        ),
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
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        reference_dds_ref_level=reference_dds_ref_level,
    )

    # pooled dds file name
    pooled_dds_file_name = get_ground_truth_dds_name(reference_dds_ref_level)

    pooled_dds_file_path = (
        processed_data_path
        / "pooled_data"
        / "tcga"
        / experiment_id
        / f"{pooled_dds_file_name}.pkl"
    )
    with open(pooled_dds_file_path, "rb") as file:
        pooled_dds = pkl.load(file)

    df_cooks_distance = fl_results["cooks_distance_df"]
    pooled_dds_cooks_df = pooled_dds.to_df("cooks")

    df_cooks_distance = df_cooks_distance.loc[pooled_dds_cooks_df.index]

    pd.testing.assert_frame_equal(pooled_dds_cooks_df, df_cooks_distance)


class CooksDistanceTester(
    UnitTester,
    ComputeCookDistances,
    GetNumReplicates,
):
    """A class to implement a unit test for Cook's distances.

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

    min_mu : float
        The minimum value of the mean expression to be considered as a valid gene.
        Used to compute the hat diagonals matrix, which is a required input for the
        computation of Cook's distances. (default: 0.5).

    trimmed_mean_num_iter: int
        The number of iterations to use when computing the trimmed mean
        in a federated way, i.e. the number of dichotomy steps.
    """

    def __init__(
        self,
        design_factors: str | list[str],
        ref_levels: dict[str, str] | None = None,
        continuous_factors: list[str] | None = None,
        reference_data_path: str | Path | None = None,
        reference_dds_ref_level: tuple[str, ...] | None = None,
        min_mu: float = 0.5,
        trimmed_mean_num_iter: int = 40,
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
            min_mu=min_mu,
            trimmed_mean_num_iter=trimmed_mean_num_iter,
        )

        self.min_mu = 0.5
        self.trimmed_mean_num_iter = trimmed_mean_num_iter

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

        #### Compute the number of replicates ####

        local_states, round_idx = self.get_num_replicates(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=clean_models,
        )

        (
            local_states,
            dispersion_for_cook_shared_state,
            round_idx,
        ) = self.compute_cook_distance(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models,
        )

        self.save_cook_distance(
            train_data_nodes,
            aggregation_node,
            local_states,
            dispersion_for_cook_shared_state,
            round_idx,
            clean_models=clean_models,
        )

    def save_cook_distance(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        dispersion_for_cook_shared_state,
        round_idx,
        clean_models,
    ):
        """Save Cook's distances. It must be used in the main pipeline while testing.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.
        aggregation_node : AggregationNode
            The aggregation node.
        local_states : dict
            Local states. Required to propagate intermediate results.
        dispersion_for_cook_shared_state : dict
            Shared state with the dispersion values for Cook's distances, in a
            "cooks_dispersions" key.
        round_idx : int
            Index of the current round.
        clean_models : bool
            Whether to clean the models after the computation.

        Returns
        -------
        local_states : dict
            Local states. Required to propagate intermediate results.
        shared_states : dict
            Shared states. Required to propagate intermediate results.
        round_idx : int
            The updated round index.
        """
        local_states, shared_states, round_idx = local_step(
            local_method=self.get_loc_cook_distance,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=dispersion_for_cook_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Get cook distances",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        aggregation_step(
            aggregation_method=self.agg_cook_distance,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Aggregate cook distances",
            round_idx=round_idx,
            clean_models=False,
        )

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def get_loc_cook_distance(self, data_from_opener, shared_state: dict) -> dict:
        """Save Cook's distances.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict
            Not used.

        Returns
        -------
        dict
            Dictionary with the following key:
            - cooks_distance_df: Cook's distances in a df
        """
        return {"cooks_distance_df": self.local_adata.to_df("cooks")}

    @remote
    @log_remote
    def agg_cook_distance(self, shared_states: list[dict]):
        """Aggregate Cook's distances.

        Parameters
        ----------
        shared_states : list[dict]
            List of shared states with the following key:
            - cooks_distance_df: Cook's distances in a df
        """
        cooks_distance_df = pd.concat(
            [shared_state["cooks_distance_df"] for shared_state in shared_states],
            axis=0,
        )
        self.results = {"cooks_distance_df": cooks_distance_df}
