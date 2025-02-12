import pickle as pkl
from pathlib import Path
from typing import Any
from typing import Literal

import anndata as ad
import numpy as np
import pytest
from fedpydeseq2_datasets.utils import get_experiment_id
from fedpydeseq2_datasets.utils import get_ground_truth_dds_name
from pydeseq2.ds import DeseqStats
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.remote import remote_data

from fedpydeseq2.core.deseq2_core.deseq2_stats.wald_tests import RunWaldTests
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import build_contrast
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote_data
from fedpydeseq2.core.utils.pass_on_results import AggPassOnResults
from tests.tcga_testing_pipe import run_tcga_testing_pipe
from tests.unit_tests.unit_test_helpers.levels import make_reference_and_fl_ref_levels
from tests.unit_tests.unit_test_helpers.unit_tester import UnitTester


@pytest.mark.parametrize(
    "design_factors, continuous_factors, contrast",
    [
        ("stage", None, None),
        (["stage", "gender"], None, ["stage", "Advanced", "Non-advanced"]),
        (["stage", "gender", "CPE"], ["CPE"], ["CPE", "", ""]),
        (["stage", "gender", "CPE"], ["CPE"], ["gender", "female", "male"]),
    ],
)
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_wald_tests_contrasts_on_small_genes(
    design_factors,
    continuous_factors,
    contrast,
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
):
    wald_tests_testing_pipe(
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
        contrast=contrast,
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
    )


@pytest.mark.parametrize(
    "design_factors, continuous_factors, alt_hypothesis",
    [
        ("stage", None, "greaterAbs"),
        (["stage", "gender"], None, "lessAbs"),
        (["stage", "gender", "CPE"], ["CPE"], "greater"),
        (["stage", "gender", "CPE"], ["CPE"], "less"),
    ],
)
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_wald_tests_alt_on_small_genes(
    design_factors,
    continuous_factors,
    alt_hypothesis,
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
):
    wald_tests_testing_pipe(
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
        alt_hypothesis=alt_hypothesis,
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
    )


@pytest.mark.self_hosted_fast
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_wald_tests_on_small_samples_on_self_hosted_fast(
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
):
    wald_tests_testing_pipe(
        raw_data_path,
        processed_data_path=processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
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


@pytest.mark.parametrize(
    "design_factors, continuous_factors, contrast",
    [
        ("stage", None, None),
        (["stage", "gender"], None, ["stage", "Advanced", "Non-advanced"]),
        (["stage", "gender", "CPE"], ["CPE"], ["CPE", "", ""]),
    ],
)
@pytest.mark.self_hosted_slow
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_wald_tests_on_self_hosted_slow(
    design_factors,
    continuous_factors,
    contrast,
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
):
    wald_tests_testing_pipe(
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
        contrast=contrast,
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
    )


def wald_tests_testing_pipe(
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
    contrast: list[str] | None = None,
    lfc_null: float = 0.0,
    alt_hypothesis: Literal["greaterAbs", "lessAbs", "greater", "less"] | None = None,
    ref_levels: dict[str, str] | None = {"stage": "Advanced"},  # noqa: B006
    reference_dds_ref_level: tuple[str, ...] | None = None,
):
    """Perform a unit test for Wald tests.

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

    continuous_factors: list or None
        The continuous factors to use.

    contrast: list or None
        The contrast to use.

    lfc_null: float
        The null hypothesis for the LFC.

    alt_hypothesis: str or None
        The alternative hypothesis.

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
    # Get FL results.
    fl_results = run_tcga_testing_pipe(
        WaldTestTester(
            design_factors=design_factors,
            continuous_factors=continuous_factors,
            contrast=contrast,
            lfc_null=lfc_null,
            alt_hypothesis=alt_hypothesis,
            ref_levels=ref_levels,
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

    # Avoid outliers not refit warning
    pooled_dds.refit_cooks = False

    # Run pydeseq2 Wald tests on the reference data
    pooled_ds = DeseqStats(
        pooled_dds,
        cooks_filter=False,
        independent_filter=False,
        contrast=contrast,
        lfc_null=lfc_null,
        alt_hypothesis=alt_hypothesis,
    )
    pooled_ds.run_wald_test()

    # Show max of the absolute difference between the results
    assert np.allclose(
        fl_results["p_values"],
        pooled_ds.p_values,
        equal_nan=True,
        rtol=0.02,
    )

    # Test the dispersion prior
    assert np.allclose(
        fl_results["wald_statistics"],
        pooled_ds.statistics,
        equal_nan=True,
        rtol=0.02,
    )

    assert np.allclose(
        fl_results["wald_se"],
        pooled_ds.SE,
        equal_nan=True,
        rtol=0.02,
    )


class WaldTestTester(UnitTester, RunWaldTests, AggPassOnResults):
    """A class to implement a unit test for Wald tests.

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

    contrast : list or None
        A list of three strings, in the following format:
        ``['variable_of_interest', 'tested_level', 'ref_level']``.
        Names must correspond to the metadata data passed to the DeseqDataSet.
        E.g., ``['condition', 'B', 'A']`` will measure the LFC of 'condition B' compared
        to 'condition A'.
        For continuous variables, the last two strings should be left empty, e.g.
        ``['measurement', '', ''].``
        If None, the last variable from the design matrix is chosen
        as the variable of interest, and the reference level is picked alphabetically.
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
    init_local_states
        A remote_data method to copy the reference dds to the local state, add the
        non_zero mask, and set the contrast in the uns attribute.

    build_compute_plan
        Build the computation graph to test Wald test computations.
    """

    def __init__(
        self,
        design_factors: str | list[str],
        ref_levels: dict[str, str] | None = None,
        continuous_factors: list[str] | None = None,
        contrast: list[str] | None = None,
        lfc_null: float = 0.0,
        alt_hypothesis: Literal["greaterAbs", "lessAbs", "greater", "less"]
        | None = None,
        joblib_backend: str = "loky",
        irls_batch_size: int = 100,
        num_jobs: int = 8,
        joblib_verbosity: int = 3,
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
            contrast=contrast,
            lfc_null=lfc_null,
            alt_hypothesis=alt_hypothesis,
            joblib_backend=joblib_backend,
            irls_batch_size=irls_batch_size,
            num_jobs=num_jobs,
            joblib_verbosity=joblib_verbosity,
            reference_data_path=reference_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
        )

        self.joblib_verbosity = joblib_verbosity
        self.num_jobs = num_jobs
        self.joblib_backend = joblib_backend
        self.irls_batch_size = irls_batch_size

        self.lfc_null = lfc_null
        self.alt_hypothesis = alt_hypothesis

        self.layers_to_save_on_disk = {"local_adata": ["cooks"], "refit_adata": None}

    def build_compute_plan(
        self,
        train_data_nodes: list[TrainDataNode],
        aggregation_node: AggregationNode,
        evaluation_strategy=None,
        num_rounds=None,
        clean_models=True,
    ):
        """Build the computation graph to test Wald test computations.

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

        local_states, _, round_idx = local_step(
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

        #### Run Wald tests ####
        local_states, wald_shared_state, round_idx = self.run_wald_tests(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            round_idx=round_idx,
            clean_models=clean_models,
        )

        aggregation_step(
            aggregation_method=self.pass_on_results,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=[wald_shared_state],
            description="Save first shared state",
            round_idx=round_idx,
            clean_models=False,
        )

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def init_local_states(self, data_from_opener: ad.AnnData, shared_state: Any):
        """Copy the reference dds to the local state and add the non_zero mask.

        Set the contrast in the uns attribute.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.
        shared_state : Any
            Shared state with a "genewise_dispersions" key.
        """

        self.local_adata = self.local_reference_dds.copy()
        # Delete the unused "_mu_hat" layer as it will raise errors
        del self.local_adata.layers["_mu_hat"]

        self.local_adata.varm["non_zero"] = self.local_reference_dds.varm["non_zero"]
        self.local_adata.uns["contrast"] = build_contrast(
            design_factors=self.design_factors,
            design_columns=self.local_adata.obsm["design_matrix"].columns,
            continuous_factors=self.continuous_factors,
            contrast=self.contrast,
        )
