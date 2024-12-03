import pickle as pkl
from pathlib import Path

import pandas as pd
import pytest
from fedpydeseq2_datasets.utils import get_experiment_id
from fedpydeseq2_datasets.utils import get_ground_truth_dds_name
from pandas.testing import assert_frame_equal
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.deseq2_core.build_design_matrix import BuildDesignMatrix
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
def test_build_design_on_small_genes(
    design_factors,
    continuous_factors,
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
):
    build_design_matrix_testing_pipe(
        raw_data_path,
        processed_data_path=local_processed_data_path,
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
        (["stage", "gender", "CPE"], ["CPE"]),
    ],
)
@pytest.mark.self_hosted_fast
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
def test_build_design_on_small_samples_on_self_hosted_fast(
    design_factors,
    continuous_factors,
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
):
    build_design_matrix_testing_pipe(
        raw_data_path,
        processed_data_path=tmp_processed_data_path,
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
        (["stage", "gender", "CPE"], ["CPE"]),
    ],
)
@pytest.mark.self_hosted_slow
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
def test_build_design_on_self_hosted_slow(
    design_factors,
    continuous_factors,
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
):
    build_design_matrix_testing_pipe(
        raw_data_path,
        processed_data_path=tmp_processed_data_path,
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


@pytest.mark.parametrize(
    "design_factors, continuous_factors",
    [
        ("stage", None),
        (["stage", "gender"], None),
        (["stage", "gender", "CPE"], ["CPE"]),
    ],
)
@pytest.mark.local
@pytest.mark.usefixtures(
    "raw_data_path", "local_processed_data_path", "tcga_assets_directory"
)
def test_build_design_on_local(
    design_factors,
    continuous_factors,
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
):
    build_design_matrix_testing_pipe(
        raw_data_path,
        processed_data_path=local_processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        small_samples=False,
        small_genes=True,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
        design_factors=design_factors,
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=("stage", "Advanced"),
        continuous_factors=continuous_factors,
    )


def build_design_matrix_testing_pipe(
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
    reference_dds_ref_level: tuple[str, ...] | None = ("stage", "Advanced"),
):
    """Perform a unit test for Wald tests
    Starting with the dispersions and LFC as the reference DeseqDataSet, perform Wald
    tests and compare the results with the reference.
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
    continuous_factors: list[str] or None
        The continuous factors amongst the design factors
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

    # pooled dds file name
    pooled_dds_file_name = get_ground_truth_dds_name(reference_dds_ref_level)

    pooled_dds_file_dir = processed_data_path / "pooled_data" / "tcga" / experiment_id

    pooled_dds_file_path = pooled_dds_file_dir / f"{pooled_dds_file_name}.pkl"

    # Get FL results.
    fl_results = run_tcga_testing_pipe(
        DesignMatrixTester(
            design_factors=design_factors,
            continuous_factors=continuous_factors,
            ref_levels=complete_ref_levels,
            reference_data_path=reference_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
            reference_pooled_data_path=pooled_dds_file_dir,
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

    fl_design_matrix = fl_results["design_matrix"]

    with open(pooled_dds_file_path, "rb") as file:
        pooled_dds = pkl.load(file)

    assert_frame_equal(
        fl_design_matrix.sort_index().reindex(
            pooled_dds.obsm["design_matrix"].columns, axis=1
        ),
        pooled_dds.obsm["design_matrix"].sort_index(),
        check_dtype=False,
    )


class DesignMatrixTester(
    UnitTester,
    BuildDesignMatrix,
):
    """A class to implement a unit test for the design matrix curve.

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
        Build the computation graph to run the creation of the design matrix and
        the means to save it.

    save_design_matrix
        Save the design matrix computed using Substra.

    get_local_design_matrix
        Get the local design matrix from the obsm of the AnnData.

    concatenate_design_matrices
        Concatenate design matrices together for registration.

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

        #### Build design matrices ####

        local_states, _, round_idx = self.build_design_matrix(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=clean_models,
        )

        print("Finished building design matrices.")

        #### Check the design matrices ####

        _ = self.save_design_matrix(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models,
        )

    def save_design_matrix(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        round_idx,
        clean_models,
    ):
        """Check the design matrix.

        Parameters
        ----------
        train_data_nodes: list
           List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: dict
            Local states. Required to propagate intermediate results.

        round_idx: int
            The current round

        clean_models: bool
            Whether to clean the models after the computation.

        Returns
        -------
        local_states: dict
            The local states, containing what is needed for evaluation.

        round_idx: int
            The updated round.

        """
        # ---- Concatenate local design matrices ----#

        local_states, shared_states, round_idx = local_step(
            local_method=self.get_local_design_matrix,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            round_idx=round_idx,
            input_local_states=local_states,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Get the local design matrix",
            clean_models=clean_models,
        )

        aggregation_step(
            aggregation_method=self.concatenate_design_matrices,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            round_idx=round_idx,
            description="Concatenating local design matrices",
            clean_models=False,
        )

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def get_local_design_matrix(
        self,
        data_from_opener,
        shared_state,
    ) -> dict:
        """
        Get the local design matrix from the obsm of the AnnData.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict or None
            Should be None.

        Returns
        -------
        dict
            A dictionary containing the local design matrix in the
            "local_design_matrix" field.
        """

        return {"local_design_matrix": self.local_adata.obsm["design_matrix"]}

    @remote
    @log_remote
    def concatenate_design_matrices(self, shared_states):
        """Concatenate design matrices together for registration.

        Parameters
        ----------
        shared_states : list
            List of design matrices from training nodes.

        """
        tot_design = pd.concat(
            [state["local_design_matrix"] for state in shared_states]
        )
        self.results = {"design_matrix": tot_design}
