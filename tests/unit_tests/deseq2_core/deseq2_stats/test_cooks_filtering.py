import pickle as pkl
from pathlib import Path
from typing import Any
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from fedpydeseq2_datasets.utils import get_experiment_id
from fedpydeseq2_datasets.utils import get_ground_truth_dds_name
from pydeseq2.ds import DeseqStats
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.deseq2_core.deseq2_stats.cooks_filtering import CooksFiltering
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data
from fedpydeseq2.core.utils.pass_on_results import AggPassOnResults
from tests.tcga_testing_pipe import run_tcga_testing_pipe
from tests.unit_tests.unit_test_helpers.levels import make_reference_and_fl_ref_levels
from tests.unit_tests.unit_test_helpers.unit_tester import UnitTester


@pytest.mark.usefixtures(
    "raw_data_path", "local_processed_data_path", "tcga_assets_directory"
)
@pytest.mark.parametrize(
    "cooks_filter, refit_cooks, design_factors, continuous_factors",
    [
        (True, False, "stage", None),
        (True, True, "stage", None),
        (False, False, "stage", None),
        (False, True, "stage", None),
        (True, False, ["stage", "gender"], None),
        (True, False, ["stage", "gender", "CPE"], ["CPE"]),
        (True, True, ["stage", "gender", "CPE"], ["CPE"]),
        (False, False, ["stage", "gender", "CPE"], ["CPE"]),
        (False, True, ["stage", "gender", "CPE"], ["CPE"]),
    ],
)
def test_cooks_filtering_small_genes(
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
    cooks_filter: bool,
    refit_cooks: bool,
    design_factors: str | list[str],
    continuous_factors: list[str] | None,
):
    cooks_filtering_testing_pipe(
        raw_data_path,
        processed_data_path=local_processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        cooks_filter=cooks_filter,
        refit_cooks=refit_cooks,
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


@pytest.mark.self_hosted_slow
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
@pytest.mark.parametrize("cooks_filter", [True, False])
def test_cooks_filtering_on_self_hosted(
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
    cooks_filter: bool,
):
    cooks_filtering_testing_pipe(
        raw_data_path,
        processed_data_path=tmp_processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        cooks_filter=cooks_filter,
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
    "raw_data_path", "local_processed_data_path", "tcga_assets_directory"
)
@pytest.mark.parametrize("cooks_filter", [True, False])
def test_cooks_filtering_on_local(
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
    cooks_filter: bool,
):
    cooks_filtering_testing_pipe(
        raw_data_path,
        processed_data_path=local_processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        cooks_filter=cooks_filter,
        small_samples=False,
        small_genes=False,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
        design_factors="stage",
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
    )


def cooks_filtering_testing_pipe(  # TODO we will have to add a case when cooks
    # TODO are refitted
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
    dataset_name="TCGA-LUAD",
    cooks_filter: bool = True,
    refit_cooks: bool = False,
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
    cooks_filter: bool
        Whether to filter Cook's distances. (default: ``True``).
    refit_cooks: bool
        Whether to refit Cook's outliers. (default: ``False``).
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
    pooled_dds_file_name = get_ground_truth_dds_name(
        reference_dds_ref_level, refit_cooks=refit_cooks
    )
    pooled_dds_file_dir = processed_data_path / "pooled_data" / "tcga" / experiment_id

    pooled_dds_file_path = pooled_dds_file_dir / f"{pooled_dds_file_name}.pkl"

    # Get FL results.
    fl_results = run_tcga_testing_pipe(
        CooksFilteringTester(
            design_factors=design_factors,
            continuous_factors=continuous_factors,
            ref_levels=complete_ref_levels,
            reference_data_path=reference_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
            cooks_filter=cooks_filter,
            refit_cooks=refit_cooks,
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
        refit_cooks=refit_cooks,
    )

    with open(pooled_dds_file_path, "rb") as file:
        pooled_dds = pkl.load(file)

    # Avoid outliers not refit warning
    if not refit_cooks:
        pooled_dds.refit_cooks = False

    # Run pydeseq2 Wald tests on the reference data
    pooled_ds = DeseqStats(
        pooled_dds, cooks_filter=cooks_filter, independent_filter=False
    )
    pooled_ds.run_wald_test()
    if cooks_filter:
        pooled_ds._cooks_filtering()

    assert np.allclose(
        fl_results["p_values"],
        pooled_ds.p_values,
        equal_nan=True,
        rtol=0.02,
    )


class CooksFilteringTester(UnitTester, CooksFiltering, AggPassOnResults):
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
    cooks_filter : bool
        Whether to filter Cook's distances. (default: ``True``).
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
    min_mu : float
        The minimum value of mu. (default: ``0.5``). Needed to compute the
        Cook's distances.
    """

    def __init__(
        self,
        design_factors: str | list[str],
        ref_levels: dict[str, str] | None = None,
        continuous_factors: list[str] | None = None,
        contrast: list[str] | None = None,
        cooks_filter: bool = True,
        refit_cooks: bool = False,
        lfc_null: float = 0.0,
        alt_hypothesis: Literal["greaterAbs", "lessAbs", "greater", "less"]
        | None = None,
        joblib_backend: str = "loky",
        irls_batch_size: int = 100,
        num_jobs: int = 8,
        joblib_verbosity: int = 3,
        reference_data_path: str | Path | None = None,
        reference_pooled_data_path: str | Path | None = None,
        reference_dds_ref_level: tuple[str, ...] | None = None,
        min_mu: float = 0.5,
        *args,
        **kwargs,
    ):
        # Add all arguments to super init so that they can be retrieved by nodes.
        super().__init__(
            design_factors=design_factors,
            ref_levels=ref_levels,
            continuous_factors=continuous_factors,
            cooks_filter=cooks_filter,
            refit_cooks=refit_cooks,
            contrast=contrast,
            lfc_null=lfc_null,
            alt_hypothesis=alt_hypothesis,
            joblib_backend=joblib_backend,
            irls_batch_size=irls_batch_size,
            num_jobs=num_jobs,
            joblib_verbosity=joblib_verbosity,
            reference_data_path=reference_data_path,
            reference_pooled_data_path=reference_pooled_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
            min_mu=min_mu,
        )

        self.joblib_verbosity = joblib_verbosity
        self.num_jobs = num_jobs
        self.joblib_backend = joblib_backend
        self.irls_batch_size = irls_batch_size

        self.lfc_null = lfc_null
        self.alt_hypothesis = alt_hypothesis

        self.reference_pooled_data_path = reference_pooled_data_path
        self.reference_dds_ref_level = reference_dds_ref_level

        self.cooks_filter = cooks_filter
        self.refit_cooks = refit_cooks

        self.min_mu = min_mu
        self.layers_to_save_on_disk = {"local_adata": ["cooks"], "refit_adata": None}

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

        #### Compute the cooks dispersions
        if self.cooks_filter:
            local_states, wald_test_shared_state, round_idx = self.cooks_filtering(
                train_data_nodes,
                aggregation_node,
                local_states,
                wald_test_shared_state,
                round_idx,
                clean_models=clean_models,
            )

        #### Save the first shared state ####

        aggregation_step(
            aggregation_method=self.pass_on_results,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=[wald_test_shared_state],
            description="Save the first shared state",
            round_idx=round_idx,
            clean_models=False,
        )

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def init_local_states(
        self, data_from_opener: ad.AnnData, shared_state: Any
    ) -> dict:
        """Copy the reference dds to the local state and add the non_zero mask.

        Also sets the total number of samples in the uns attribute.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.
        shared_state : Any
            Shared state. Not used.
        """
        pooled_dds_file_name = get_ground_truth_dds_name(
            self.reference_dds_ref_level, refit_cooks=self.refit_cooks
        )

        pooled_dds_file_path = (
            Path(self.reference_pooled_data_path) / f"{pooled_dds_file_name}.pkl"
        )

        with open(pooled_dds_file_path, "rb") as file:
            pooled_dds = pkl.load(file)

        counts_by_lvl = pooled_dds.obsm["design_matrix"].value_counts()
        self.local_adata = self.local_reference_dds.copy()
        # Delete the unused "_mu_hat" layer as it will raise errors
        del self.local_adata.layers["_mu_hat"]
        if "replace_cooks" in self.local_adata.layers.keys():
            del self.local_adata.layers["replace_cooks"]

        self.local_adata.varm["non_zero"] = self.local_reference_dds.varm["non_zero"]
        self.local_adata.uns["tot_num_samples"] = pooled_dds.n_obs
        self.local_adata.uns["num_replicates"] = pd.Series(counts_by_lvl.values)
        self.local_adata.obs["cells"] = [
            np.argwhere(counts_by_lvl.index == tuple(design))[0, 0]
            for design in self.local_adata.obsm["design_matrix"].values
        ]
        self.local_adata.uns["n_params"] = self.local_adata.obsm["design_matrix"].shape[
            1
        ]
        return {}

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
        pooled_dds_file_name = get_ground_truth_dds_name(
            self.reference_dds_ref_level, refit_cooks=self.refit_cooks
        )

        pooled_dds_file_path = (
            Path(self.reference_pooled_data_path) / f"{pooled_dds_file_name}.pkl"
        )

        with open(pooled_dds_file_path, "rb") as file:
            pooled_dds = pkl.load(file)

        # Avoid outliers not refit warning
        if not self.refit_cooks:
            pooled_dds.refit_cooks = False

        # Run pydeseq2 Wald tests on the reference data
        pooled_ds = DeseqStats(pooled_dds, cooks_filter=False, independent_filter=False)
        pooled_ds.run_wald_test()

        return {
            "p_values": pooled_ds.p_values,
            "wald_statistics": pooled_ds.statistics,
            "wald_se": pooled_ds.SE,
        }
