import pickle as pkl
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from fedpydeseq2_datasets.utils import get_experiment_id
from fedpydeseq2_datasets.utils import get_ground_truth_dds_name
from pydeseq2.ds import DeseqStats
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.deseq2_core.deseq2_stats.compute_padj import (
    ComputeAdjustedPValues,
)
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data
from fedpydeseq2.core.utils.pass_on_results import AggPassOnResults
from tests.tcga_testing_pipe import run_tcga_testing_pipe
from tests.unit_tests.unit_test_helpers.levels import make_reference_and_fl_ref_levels
from tests.unit_tests.unit_test_helpers.unit_tester import UnitTester


@pytest.mark.parametrize(
    "design_factors, continuous_factors, independent_filter",
    [
        ("stage", None, True),
        (["stage", "gender"], None, True),
        (["stage", "gender", "CPE"], ["CPE"], True),
        ("stage", None, False),
        (["stage", "gender"], None, False),
        (["stage", "gender", "CPE"], ["CPE"], False),
    ],
)
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_compute_adj_small_genes(
    design_factors,
    continuous_factors,
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
    independent_filter: bool,
):
    compute_padj_testin_pipe(
        raw_data_path,
        processed_data_path=processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        independent_filter=independent_filter,
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


@pytest.mark.self_hosted_fast
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
@pytest.mark.parametrize("independent_filter", [True, False])
def test_compute_adj_small_samples(
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
    independent_filter: bool,
):
    compute_padj_testin_pipe(
        raw_data_path,
        processed_data_path=processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        independent_filter=independent_filter,
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
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
@pytest.mark.parametrize("independent_filter", [True, False])
def test_compute_adj_on_self_hosted(
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
    independent_filter: bool,
):
    compute_padj_testin_pipe(
        raw_data_path,
        processed_data_path=processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        independent_filter=independent_filter,
        small_samples=False,
        small_genes=False,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
        design_factors="stage",
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
    )


@pytest.mark.local
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
@pytest.mark.parametrize("independent_filter", [True, False])
def test_compute_adj_on_local(
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
    independent_filter: bool,
):
    compute_padj_testin_pipe(
        raw_data_path,
        processed_data_path=processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        independent_filter=independent_filter,
        small_samples=False,
        small_genes=True,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
        design_factors="stage",
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
    )


def compute_padj_testin_pipe(
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
    dataset_name="TCGA-LUAD",
    independent_filter=True,
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
    """Perform a unit test for Wald tests Starting with the dispersions and LFC as the
    reference DeseqDataSet, perform Wald tests and compare the results with the
    reference.

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
    independent_filter: bool
        Whether to use independent filtering to correct the p-values trend.
        (default: ``True``).
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

    # pooled dds file name
    pooled_dds_file_name = get_ground_truth_dds_name(reference_dds_ref_level)

    pooled_dds_file_dir = processed_data_path / "pooled_data" / "tcga" / experiment_id

    pooled_dds_file_path = pooled_dds_file_dir / f"{pooled_dds_file_name}.pkl"

    # Get FL results.
    fl_results = run_tcga_testing_pipe(
        ComputeAdjustedPValuesTester(
            design_factors=design_factors,
            continuous_factors=continuous_factors,
            ref_levels=ref_levels,
            reference_data_path=reference_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
            independent_filter=independent_filter,
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

    with open(pooled_dds_file_path, "rb") as file:
        pooled_dds = pkl.load(file)

    # Avoid outliers not refit warning
    pooled_dds.refit_cooks = False

    # Run pydeseq2 Wald tests on the reference data
    pooled_ds = DeseqStats(
        pooled_dds, cooks_filter=False, independent_filter=independent_filter
    )
    pooled_ds.summary()

    assert np.allclose(
        fl_results["padj"],
        pooled_ds.padj,
        equal_nan=True,
        rtol=0.02,
    )


class ComputeAdjustedPValuesTester(
    UnitTester, ComputeAdjustedPValues, AggPassOnResults
):
    """A class to implement a for p-value adjustment.

    # TODO merge the method for running the Wald test on the reference data with the
    # TODO equivalent method for testing cooks.


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
    independent_filter : bool
        Whether to use independent filtering to correct the p-values trend.
        (default: ``True``).
    alpha : float
        P-value and adjusted p-value significance threshold (usually 0.05).
        (default: ``0.05``).
    reference_data_path : str or Path
        The path to the reference data. This is used to build the reference
        DeseqDataSet. This is only used for testing purposes, and should not be
        used in a real-world scenario.
    reference_pooled_data_path : str or Path
        The path to the reference pooled data. This is used to build the reference
        DeseqStats object. This is only used for testing purposes, and should not be
        used in a real-world scenario.
    reference_dds_ref_level : tuple por None
        The reference level of the reference DeseqDataSet. This is used to build the
        reference DeseqDataSet. This is only used for testing purposes, and should not
        be used in a real-world scenario.
    """

    def __init__(
        self,
        design_factors: str | list[str],
        ref_levels: dict[str, str] | None = None,
        continuous_factors: list[str] | None = None,
        contrast: list[str] | None = None,
        independent_filter: bool = True,
        alpha: float = 0.05,
        lfc_null: float = 0.0,
        alt_hypothesis: (
            Literal["greaterAbs", "lessAbs", "greater", "less"] | None
        ) = None,
        joblib_backend: str = "loky",
        irls_batch_size: int = 100,
        num_jobs: int = 8,
        joblib_verbosity: int = 3,
        reference_data_path: str | Path | None = None,
        reference_pooled_data_path: str | Path | None = None,
        reference_dds_ref_level: tuple[str, ...] | None = None,
        *args,
        **kwargs,
    ):
        # Add all arguments to super init so that they can be retrieved by nodes.
        super().__init__(
            design_factors=design_factors,
            ref_levels=ref_levels,
            continuous_factors=continuous_factors,
            independent_filter=independent_filter,
            alpha=alpha,
            contrast=contrast,
            lfc_null=lfc_null,
            alt_hypothesis=alt_hypothesis,
            joblib_backend=joblib_backend,
            irls_batch_size=joblib_backend,
            num_jobs=num_jobs,
            joblib_verbosity=joblib_verbosity,
            reference_data_path=reference_data_path,
            reference_pooled_data_path=reference_pooled_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
        )

        self.joblib_verbosity = joblib_verbosity
        self.num_jobs = num_jobs
        self.joblib_backend = joblib_backend
        self.irls_batch_size = irls_batch_size

        self.lfc_null = lfc_null
        self.alt_hypothesis = alt_hypothesis

        self.reference_pooled_data_path = reference_pooled_data_path
        self.reference_dds_ref_level = reference_dds_ref_level

        self.independent_filter = independent_filter
        self.alpha = alpha

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

        local_states, empty_shared_states, round_idx = local_step(
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

        #### Compute the reference wald test results ####
        wald_test_shared_state, round_idx = aggregation_step(
            aggregation_method=self.agg_run_wald_test_on_ground_truth,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=empty_shared_states,
            description="Run Wald test on ground truth",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        #### Compute the adusted p-values ####
        (
            local_states,
            round_idx,
        ) = self.compute_adjusted_p_values(
            train_data_nodes,
            aggregation_node,
            local_states,
            wald_test_shared_state,
            round_idx,
            clean_models=clean_models,
        )

        local_states, wald_test_shared_states, round_idx = local_step(
            local_method=self.get_results_from_local_adata,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Get results to share from the local centers",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        aggregation_step(
            aggregation_method=self.pass_on_results,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=wald_test_shared_states,
            description="Save the first shared state.",
            round_idx=round_idx,
            clean_models=False,
        )

    @remote
    @log_remote
    def agg_run_wald_test_on_ground_truth(self, shared_states: dict) -> dict:
        """Run Wald tests on the reference data.

        Parameters
        ----------
        shared_states : dict
            Shared states. Not used.
        Returns
        -------
        shared_states : dict
            Shared states. The new shared state contains the Wald test results on the
            pooled reference.
        """
        pooled_dds_file_name = get_ground_truth_dds_name(self.reference_dds_ref_level)

        pooled_dds_file_path = (
            Path(self.reference_pooled_data_path) / f"{pooled_dds_file_name}.pkl"
        )

        with open(pooled_dds_file_path, "rb") as file:
            pooled_dds = pkl.load(file)

        # Avoid outliers not refit warning
        pooled_dds.refit_cooks = False  # TODO to change after refit cooks implemented.

        # Run pydeseq2 Wald tests on the reference data
        pooled_ds = DeseqStats(pooled_dds, cooks_filter=False, independent_filter=False)
        pooled_ds.run_wald_test()

        return {
            "p_values": pooled_ds.p_values.to_numpy(),
            "wald_statistics": pooled_ds.statistics,
            "wald_se": pooled_ds.SE,
        }

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def get_results_from_local_adata(
        self,
        data_from_opener,
        shared_state: dict | None,
    ) -> dict:
        """Get the results to share from the local states.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict, optional
            Not used.

        Returns
        -------
        dict
            Shared state containing the adjusted p-values, the p-values, the Wald
            standard errors, and the Wald statistics.
        """

        shared_state = {
            varm_key: self.local_adata.varm[varm_key]
            for varm_key in ["padj", "p_values", "wald_se", "wald_statistics"]
        }
        return shared_state
