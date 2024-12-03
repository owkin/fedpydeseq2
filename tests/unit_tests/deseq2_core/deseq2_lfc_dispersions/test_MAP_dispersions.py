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
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_MAP_dispersions import (  # noqa: E501
    ComputeMAPDispersions,
)
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data
from fedpydeseq2.core.utils.pass_on_results import AggPassOnResults
from tests.tcga_testing_pipe import run_tcga_testing_pipe
from tests.unit_tests.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions.utils_genewise_dispersions import (  # noqa: E501
    perform_dispersions_and_nll_relative_check,
)
from tests.unit_tests.unit_test_helpers.levels import make_reference_and_fl_ref_levels
from tests.unit_tests.unit_test_helpers.pass_on_first_shared_state import (
    AggPassOnFirstSharedState,
)
from tests.unit_tests.unit_test_helpers.unit_tester import UnitTester


@pytest.mark.usefixtures(
    "raw_data_path", "local_processed_data_path", "tcga_assets_directory"
)
@pytest.mark.parametrize(
    "design_factors, continuous_factors",
    [
        ("stage", None),
        (["stage", "gender"], None),
        (["stage", "gender", "CPE"], ["CPE"]),
    ],
)
def test_MAP_dispersions_on_small_genes_small_samples(
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
    design_factors,
    continuous_factors,
):
    MAP_dispersions_testing_pipe(
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


@pytest.mark.self_hosted_fast
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
@pytest.mark.parametrize(
    "design_factors, continuous_factors, tolerated_failed_genes",
    [
        ("stage", None, 0),
        (["stage", "gender"], None, 0),
        (["stage", "gender", "CPE"], ["CPE"], 1),
    ],
)
def test_MAP_dispersions_on_small_samples_on_self_hosted_fast(
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
    design_factors,
    continuous_factors,
    tolerated_failed_genes,
):
    MAP_dispersions_testing_pipe(
        raw_data_path,
        tmp_processed_data_path,
        tcga_assets_directory,
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
        tolerated_failed_genes=tolerated_failed_genes,
    )


@pytest.mark.self_hosted_slow
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
@pytest.mark.parametrize(
    "design_factors, continuous_factors",
    [
        ("stage", None),
        (["stage", "gender"], None),
        (["stage", "gender", "CPE"], ["CPE"]),
    ],
)
def test_MAP_dispersions_on_self_hosted_slow(
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
    design_factors,
    continuous_factors,
):
    MAP_dispersions_testing_pipe(
        raw_data_path,
        tmp_processed_data_path,
        tcga_assets_directory,
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


@pytest.mark.self_hosted_slow
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
@pytest.mark.parametrize(
    "design_factors, continuous_factors",
    [
        ("stage", None),
        (["stage", "gender"], None),
    ],
)
def test_MAP_dispersions_paad_on_self_hosted_slow(
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
    design_factors,
    continuous_factors,
):
    MAP_dispersions_testing_pipe(
        raw_data_path,
        tmp_processed_data_path,
        tcga_assets_directory,
        dataset_name="TCGA-PAAD",
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


def MAP_dispersions_testing_pipe(
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
    rtol: float = 0.02,
    atol: float = 1e-3,
    nll_rtol: float = 0.02,
    nll_atol: float = 1e-3,
    tolerated_failed_genes: int = 0,
):
    """Perform a unit test for the MAP dispersions.

    Starting with the same dispersion trend curve as the reference dataset, compute MAP
    dispersions and compare the results with the reference.

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

    rtol: float
        The relative tolerance for between the FL and pooled dispersions.

    atol: float
        The absolute tolerance for between the FL and pooled dispersions.

    nll_rtol: float
        The relative tolerance for between the FL and pooled likelihoods, in the
        case of a failed dispersion check.

    nll_atol: float
        The absolute tolerance for between the FL and pooled likelihoods, in the
        case of a failed dispersion check.

    tolerated_failed_genes: int
        The number of genes that are allowed to fail the test.

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
    # Get FL results.
    fl_results = run_tcga_testing_pipe(
        MAPDispersionsTester(
            design_factors=design_factors,
            ref_levels=complete_ref_levels,
            continuous_factors=continuous_factors,
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

    # Tests that the MAP dispersions are close to the pooled ones, or if not,
    # that the adjusted log likelihood is close or better

    fl_dispersions = fl_results["MAP_dispersions"]
    perform_dispersions_and_nll_relative_check(
        fl_dispersions,
        pooled_dds,
        dispersions_param_name="MAP_dispersions",
        prior_reg=True,
        rtol=rtol,
        atol=atol,
        nll_rtol=nll_rtol,
        nll_atol=nll_atol,
        tolerated_failed_genes=tolerated_failed_genes,
    )


class MAPDispersionsTester(
    UnitTester, ComputeMAPDispersions, AggPassOnResults, AggPassOnFirstSharedState
):
    """A class to implement a unit test for the MAP dispersions.

    Note that this test checks the MAP dispersions BEFORE filtering.
    The filtering is also done in the ComputeMAPDispersions class, but is
    tested separately.

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

    min_disp : float
        Lower threshold for dispersion parameters. (default: ``1e-8``).

    max_disp : float
        Upper threshold for dispersion parameters.
        Note: The threshold that is actually enforced is max(max_disp, len(counts)).
        (default: ``10``).

    grid_batch_size : int
        The number of genes to put in each batch for local parallel processing.
        (default: ``100``).

    grid_depth : int
        The number of grid interval selections to perform (if using GridSearch).
        (default: ``3``).

    grid_length : int
        The number of grid points to use for the grid search (if using GridSearch).
        (default: ``100``).

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
    init_local_states
        A remote_data method which copies the reference dds to the local state
        and adds the non_zero mask.
        It sets the max_disp to the maximum of the reference dds number of samples
        and the max_disp parameter.
        It returns a dictionary with the number of samples in
        the "num_samples" field.

    sum_num_samples
        A remote method which computes the total number of samples to set max_disp.
        It returns a dictionary with the total number of samples in the
        "tot_num_samples" field.

    set_max_disp
        A remote_data method which sets max_disp using the total number of samples in
        the study.
        It returns a dictionary with the fitted dispersions in the "fitted_dispersions"
        field and the prior variance of the dispersions in the "prior_disp_var" field.

    get_MAP_dispersions
        A remote_data method which gets the filtered dispersions.
        It returns a dictionary with the MAP dispersions in the "MAP_dispersions" field.

    create_trend_curve_fitting_shared_state
        A method which creates the trend curve fitting shared state from reference.

    save_MAP_dispersions
        A method which saves the MAP dispersions.

    build_compute_plan
        A method which builds the computation graph to test the computation of the MAP
        dispersions.

    """

    def __init__(
        self,
        design_factors: str | list[str],
        ref_levels: dict[str, str] | None = None,
        continuous_factors: list[str] | None = None,
        min_disp: float = 1e-8,
        max_disp: float = 10.0,
        grid_batch_size: int = 250,
        grid_depth: int = 3,
        grid_length: int = 100,
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
            min_disp=min_disp,
            max_disp=max_disp,
            grid_batch_size=grid_batch_size,
            grid_depth=grid_depth,
            grid_length=grid_length,
            num_jobs=num_jobs,
            reference_data_path=reference_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
        )

        #### Define hyper parameters ####

        self.min_disp = min_disp
        self.max_disp = max_disp
        self.grid_batch_size = grid_batch_size
        self.grid_depth = grid_depth
        self.grid_length = grid_length
        self.num_jobs = num_jobs

        # Add layers to save
        self.layers_to_save_on_disk = {"local_adata": ["_mu_hat"], "refit_adata": None}

    def build_compute_plan(
        self,
        train_data_nodes: list[TrainDataNode],
        aggregation_node: AggregationNode,
        evaluation_strategy=None,
        num_rounds=None,
        clean_models=True,
    ):
        """Build the computation graph to test the computation of the MAP dispersions.

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

        #### Create trend curve fitting shared state ####

        (
            local_states,
            trend_curve_shared_state,
            round_idx,
        ) = self.create_trend_curve_fitting_shared_state(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            round_idx=round_idx,
            clean_models=clean_models,
        )

        #### Fit MAP dispersions with MLE ####

        local_states, round_idx = self.fit_MAP_dispersions(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            shared_state=trend_curve_shared_state,
            round_idx=round_idx,
            clean_models=clean_models,
        )

        self.save_MAP_dispersions(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            round_idx=round_idx,
            clean_models=clean_models,
        )

    def create_trend_curve_fitting_shared_state(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        round_idx,
        clean_models,
    ):
        """Create the trend curve fitting shared state from reference.

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

        Returns
        -------
        local_states : dict
            Dictionary of local states.

        trend_curve_shared_state : dict
            Trend curve shared state. It is a dictionary with a field
            "fitted_dispersion" containing the fitted dispersions from the trend curve,
            and a field "prior_disp_var" containing the prior variance
            of the dispersions.

        round_idx : int
            Round index.


        """
        #### Load reference dataset as local_adata and set local states ####

        local_states, init_shared_states, round_idx = local_step(
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

        ### Aggregation step to compute the total number of samples ###
        shared_state, round_idx = aggregation_step(
            aggregation_method=self.sum_num_samples,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=init_shared_states,
            description="Get the total number of samples.",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        local_states, max_disp_shared_states, round_idx = local_step(
            local_method=self.set_max_disp,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Compute max_disp and forward fitted dispersions",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        trend_curve_shared_state, round_idx = aggregation_step(
            aggregation_method=self.pass_on_shared_state,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=max_disp_shared_states,
            description="Pass on the shared state",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        return local_states, trend_curve_shared_state, round_idx

    def save_MAP_dispersions(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        round_idx,
        clean_models,
    ):
        """Save the MAP dispersions.

        This method gets the MAP dispersions from the local states and saves them in the
        results field of the local states.

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
            local_method=self.get_MAP_dispersions,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Get filtered dispersions.",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        # ---- Save the MAP dispersions ---- #

        aggregation_step(
            aggregation_method=self.pass_on_results,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Save the MAP dispersions.",
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

        Set the max_disp to the maximum of the reference dds number of samples and the
        max_disp parameter.

        Returns a dictionary with the number of samples in the "num_samples" field.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : Any
            Shared state with "fitted_dispersions" and "prior_disp_var" keys.

        Returns
        -------
        dict
            The number of samples in the "num_samples" field.
        """

        self.local_adata = self.local_reference_dds.copy()
        self.local_adata.varm["non_zero"] = self.local_reference_dds.varm["non_zero"]
        self.local_adata.uns["max_disp"] = max(
            self.max_disp, self.local_reference_dds.n_obs
        )

        return {
            "num_samples": self.local_adata.n_obs,
        }

    @remote
    @log_remote
    def sum_num_samples(self, shared_states):
        """Compute the total number of samples to set max_disp.

        Parameters
        ----------
        shared_states : list
            List of initial shared states copied from the reference adata.

        Returns
        -------
        dict

        """
        tot_num_samples = np.sum([state["num_samples"] for state in shared_states])
        return {"tot_num_samples": tot_num_samples}

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def set_max_disp(self, data_from_opener: ad.AnnData, shared_state: Any) -> dict:
        """Set max_disp using the total number of samples in the study.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : Any
            Shared state with "fitted_dispersions" and "prior_disp_var" keys.
        """

        self.local_adata.uns["max_disp"] = max(
            self.max_disp, shared_state["tot_num_samples"]
        )

        return {
            "fitted_dispersions": self.local_adata.varm["fitted_dispersions"],
            "prior_disp_var": self.local_adata.uns["prior_disp_var"],
        }

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def get_MAP_dispersions(self, data_from_opener, shared_state):
        """Get the filtered dispersions.

        Returns
        -------
        dict
            A dictionary with the MAP dispersions in the "MAP_dispersions" field.
        """
        return {"MAP_dispersions": self.local_adata.varm["MAP_dispersions"]}
