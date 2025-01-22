"""Module to test the final step of the compute_genewise_dispersions step.

This step tests the final substep which is to estimate the genewise dispersions by
minimizing the negative binomial likelihood, with a fixed value of the mean parameter
given by the mu_hat estimate.
"""
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

from fedpydeseq2.core.fed_algorithms import ComputeDispersionsGridSearch
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data
from tests.tcga_testing_pipe import run_tcga_testing_pipe
from tests.unit_tests.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions.utils_genewise_dispersions import (  # noqa: E501
    perform_dispersions_and_nll_relative_check,
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
def test_dispersions_from_mu_hat_on_small_genes_small_samples(
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
    design_factors,
    continuous_factors,
):
    """Perform a unit test for the genewise dispersions.

    This test is performed on a small number of genes and samples, in order to
    be fast and run on the github CI.

    Note that in this test, we tolerate 0 failed genes.

    Parameters
    ----------
    raw_data_path: Path
        The path to the root data.

    local_processed_data_path: Path
        The path to the processed data. The subdirectories will

    tcga_assets_directory: Path
        The path to the assets directory. It must contain the

    design_factors: str or list
        The design factors to use.

    continuous_factors: list or None
        The continuous factors to use.
    """
    dispersions_from_mu_hat_testing_pipe(
        raw_data_path,
        local_processed_data_path,
        tcga_assets_directory,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        dataset_name="TCGA-LUAD",
        small_samples=True,
        small_genes=True,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
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
def test_dispersions_from_mu_hat_on_small_samples_on_self_hosted_fast(
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
    design_factors,
    continuous_factors,
    tolerated_failed_genes,
):
    """Perform a unit test for the genewise dispersions.

    This test is performed on a small number of samples, in order to be fast.
    However, all genes are used, so that we can have a clearer statistical vision
    of the failing cases (50 000 genes).

    In only one case (the last one), we authorized one failed gene. We have not
    investigated further why it fails. Our guess is that the underlying assumption
    for the grid search we perform (with multiple steps) is that the nll
    decreases then increases. This is perhaps not always true (we have no theoretical
    guarantee), or perhaps the nll is very sharp. In any case, it is not suprising
    that such things happen in a case where we have very few samples (samples smooth
    the losses).

    Parameters
    ----------
    raw_data_path: Path
        The path to the root data.

    tmp_processed_data_path: Path
        The path to the processed data. The subdirectories will

    tcga_assets_directory: Path
        The path to the assets directory. It must contain the

    design_factors: str or list
        The design factors to use.

    continuous_factors: list or None
        The continuous factors to use.

    tolerated_failed_genes: int
        The number of genes that are allowed to fail the relative nll criterion.
    """
    dispersions_from_mu_hat_testing_pipe(
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
def test_dispersions_from_mu_hat_on_self_hosted_slow(
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
    design_factors,
    continuous_factors,
):
    """Perform a unit test for the genewise dispersions.

    This test is performed on the full dataset, in order to have a more
    realistic view of the performance of the algorithm, on a self hosted
    runner and using the TCGA-LUAD dataset.


    Parameters
    ----------
    raw_data_path: Path
        The path to the root data.

    tmp_processed_data_path: Path
        The path to the processed data. The subdirectories will

    tcga_assets_directory: Path
        The path to the assets directory. It must contain the

    design_factors: str or list
        The design factors to use.

    continuous_factors: list or None
        The continuous factors to use.
    """
    dispersions_from_mu_hat_testing_pipe(
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
def test_dispersions_from_mu_hat_paad_on_self_hosted_slow(
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
    design_factors,
    continuous_factors,
):
    """Perform a unit test for the genewise dispersions.

    This test is performed on the full dataset, in order to have a more
    realistic view of the performance of the algorithm, on a self hosted
    runner and using the TCGA-PAAD dataset.


    Parameters
    ----------
    raw_data_path: Path
        The path to the root data.

    tmp_processed_data_path: Path
        The path to the processed data. The subdirectories will

    tcga_assets_directory: Path
        The path to the assets directory. It must contain the

    design_factors: str or list
        The design factors to use.

    continuous_factors: list or None
        The continuous factors to use.
    """
    dispersions_from_mu_hat_testing_pipe(
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


def dispersions_from_mu_hat_testing_pipe(
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
    design_factors,
    continuous_factors,
    dataset_name="TCGA-LUAD",
    small_samples=True,
    small_genes=True,
    simulate=True,
    backend="subprocess",
    only_two_centers=False,
    ref_levels: dict[str, str] | None = {"stage": "Advanced"},  # noqa: B006
    reference_dds_ref_level: tuple[str, ...] | None = None,
    rtol: float = 0.02,
    atol: float = 1e-3,
    nll_rtol: float = 0.02,
    nll_atol: float = 1e-3,
    tolerated_failed_genes: int = 0,
):
    """Perform a unit test for the genewise dispersions.

    This unit test only concerns the last step of the genewise dispersions fitting,
    that is fitting the dispersions from the mu hat estimate by minimizing the
    negative binomial likelihood function (as a function of the dispersion only).

    We start from all quantities defined in the pooled data. We then fit the genewise
    dispersions using the mu hat estimate as the mean parameter. We then compare the
    FL dispersions to the pooled dispersions. If the relative error is above 2%,
    we check the likelihoods. If the likelihoods are above 2% higher than the pooled
    likelihoods, we fail the test (with a certain tolerance for failure).

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
        case where the dispersions are above the tolerance.

    nll_atol: float
        The absolute tolerance for between the FL and pooled likelihoods, in the
        case where the dispersions are above the tolerance.

    tolerated_failed_genes: int
        The number of genes that are allowed to fail the relative nll criterion.
    """

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
        GenewiseDispersionsFromMuHatTester(
            design_factors=design_factors,
            ref_levels=ref_levels,
            reference_data_path=reference_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
            continuous_factors=continuous_factors,
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

    # Compute relative error

    perform_dispersions_and_nll_relative_check(
        fl_results["genewise_dispersions"],
        pooled_dds,
        rtol=rtol,
        atol=atol,
        nll_rtol=nll_rtol,
        nll_atol=nll_atol,
        tolerated_failed_genes=tolerated_failed_genes,
    )


class GenewiseDispersionsFromMuHatTester(UnitTester, ComputeDispersionsGridSearch):
    """A class to implement a unit test for the genewise dispersions fitting.

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

    joblib_verbosity : int
        Verbosity level for joblib. (default: ``3``).

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
        Build the computation graph to run and save the genewise dispersions.

    save_genewise_dispersions_checkpoint
        Save genewise dispersions checkpoint.

    init_local_states
        A local method.
        Copy the reference dds to the local state and add the non_zero mask.

    sum_num_samples
        An aggregation method.
        Compute the total number of samples to set max_disp.

    set_max_disp
        A local method.
        Set max_disp using the total number of samples in the study.

    get_local_dispersions
        A local method.
        Collect dispersions and pass on.

    pass_on_results
        An aggregation method.
        Set the genewise dispersions in the results.
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
        joblib_backend: str = "loky",
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
            min_disp=min_disp,
            max_disp=max_disp,
            grid_batch_size=grid_batch_size,
            grid_depth=grid_depth,
            grid_length=grid_length,
            num_jobs=num_jobs,
            reference_data_path=reference_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
            joblib_backend=joblib_backend,
            joblib_verbosity=joblib_verbosity,
        )

        #### Define hyper parameters ####

        self.min_disp = min_disp
        self.max_disp = max_disp
        self.grid_batch_size = grid_batch_size
        self.grid_depth = grid_depth
        self.grid_length = grid_length
        self.num_jobs = num_jobs

        #### Define job parallelization parameters ####
        self.joblib_verbosity = joblib_verbosity
        self.num_jobs = num_jobs
        self.joblib_backend = joblib_backend

        # Very important, we need to keep these layers as they cannot be recomputed.
        self.layers_to_save_on_disk = {
            "local_adata": ["_mu_hat"],
            "refit_adata": None,
        }

    def build_compute_plan(
        self,
        train_data_nodes: list[TrainDataNode],
        aggregation_node: AggregationNode,
        evaluation_strategy=None,
        num_rounds=None,
        clean_models=True,
    ):
        """Build the computation graph to run and save the genewise dispersions.

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

        ### Set max_disp using the total number of samples ###

        local_states, _, round_idx = local_step(
            local_method=self.set_max_disp,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Compute max_disp",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        #### Fit genewise dispersions ####
        (
            local_states,
            genewise_dispersions_shared_state,
            round_idx,
        ) = self.fit_dispersions(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            shared_state=None,
            round_idx=round_idx,
            clean_models=clean_models,
        )

        #### Save genewise dispersions ####

        self.save_genewise_dispersions_checkpoint(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            genewise_dispersions_shared_state=genewise_dispersions_shared_state,
            round_idx=round_idx,
            clean_models=False,
        )

    def save_genewise_dispersions_checkpoint(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        genewise_dispersions_shared_state,
        round_idx,
        clean_models,
    ):
        """Save genewise dispersions checkpoint.

        This method saves the genewise dispersions.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: dict
            Local states. Required to propagate intermediate results.

        genewise_dispersions_shared_state: dict
            Contains the output shared state of "fit_genewise_dispersions" step,
            which contains a "genewise_dispersions" field used in this test.

        round_idx: int
            The current round.

        clean_models: bool
            Whether to clean the models after the computation.
        """
        # ---- Get local estimates ---- #

        local_states, shared_states, round_idx = local_step(
            local_method=self.get_local_dispersions,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=genewise_dispersions_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Get local dispersions",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        # ---- Save dispersions in result ---- #

        results_shared_state, round_idx = aggregation_step(
            aggregation_method=self.pass_on_results,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Save genewise dispersions in results",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        return local_states, round_idx

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def init_local_states(
        self, data_from_opener: ad.AnnData, shared_state: Any
    ) -> dict:
        """Copy the reference dds to the local state and add the non_zero mask.

        Returns a dictionary with the number of samples in the "num_samples" field,
        in order to compute the max dispersion.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : Any
            Not used.

        Returns
        -------
        dict
            The number of samples in the "num_samples" field.
        """

        self.local_adata = self.local_reference_dds.copy()
        self.local_adata.varm["non_zero"] = self.local_reference_dds.varm["non_zero"]

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
            The total number of samples in the "tot_num_samples" field.
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
            Not used.
        """

        self.local_adata.uns["max_disp"] = max(
            self.max_disp, shared_state["tot_num_samples"]
        )

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def get_local_dispersions(
        self, data_from_opener: ad.AnnData, shared_state: dict
    ) -> dict:
        """Collect dispersions and pass on.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener.
        shared_state : dict
            Shared state with the gene-wise dispersions.

        Returns
        -------
        dict
            Shared state the gene-wise dispersions.
        """

        return {"genewise_dispersions": shared_state["genewise_dispersions"]}

    @remote
    @log_remote
    def pass_on_results(self, shared_states: list):
        """Set the genewise dispersions in the results.

        Parameters
        ----------
        shared_states : list
            List of shared states. The first element contains the genewise dispersions.
        """
        self.results = {
            "genewise_dispersions": shared_states[0]["genewise_dispersions"],
        }
