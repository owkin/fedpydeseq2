"""Module to test the genewise dispersions computation in a single factor case."""

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

from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions import (  # noqa: E501
    ComputeGenewiseDispersions,
)
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
    "design_factors",
    [
        "stage",
    ],
)
def test_genewise_dispersions_on_small_genes_small_samples(
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
    design_factors,
):
    genewise_dispersions_testing_pipe(
        raw_data_path,
        local_processed_data_path,
        tcga_assets_directory,
        design_factors=design_factors,
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
    "design_factors",
    [
        "stage",
    ],
)
def test_genewise_dispersions_on_small_samples_on_self_hosted_fast(
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
    design_factors,
):
    genewise_dispersions_testing_pipe(
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
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
    )


@pytest.mark.self_hosted_slow
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
@pytest.mark.parametrize("design_factors", ["stage"])
def test_genewise_dispersions_on_self_hosted_slow(
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
    design_factors,
):
    genewise_dispersions_testing_pipe(
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
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
    )


def genewise_dispersions_testing_pipe(
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
    design_factors,
    dataset_name="TCGA-LUAD",
    small_samples=True,
    small_genes=True,
    simulate=True,
    backend="subprocess",
    only_two_centers=False,
    ref_levels: dict[str, str] | None = {"stage": "Advanced"},  # noqa: B006
    reference_dds_ref_level: tuple[str, ...] | None = None,
    rtol=0.02,
    atol=1e-3,
    nll_rtol=0.02,
    nll_atol=1e-3,
    tolerated_failed_genes=0,
):
    """Perform a unit test for the genewise dispersions.

    This steps tests the three quantities generated by the genewise dispersions fitting:
    - the genewise dispersions themselves which are the endpoint;
    - the method of moments dispersions (MoM) which are the starting point;
    - the initial estimate of mu, the mean of the counts, which is an intermediate step.


    Starting with the same size factors as the reference dataset, compute genewise
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

    nll_atol: float
        The absolute tolerance for between the FL and pooled likelihoods, in the

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
        continuous_factors=None,
    )

    reference_data_path = (
        local_processed_data_path / "centers_data" / "tcga" / experiment_id
    )
    # Get FL results.
    fl_results = run_tcga_testing_pipe(
        GenewiseDispersionsTester(
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

    # Check that the MoM dispersions are close to the pooled ones

    assert np.allclose(
        fl_results["MoM_dispersions"],
        pooled_dds.varm["_MoM_dispersions"],
        equal_nan=True,
    )

    # get sample ids for the pooled dds
    sample_ids = pooled_dds.obs_names
    sample_ids_fl = fl_results["sample_ids"]
    # Get the permutation that sorts the sample ids
    perm = np.argsort(sample_ids)
    perm_fl = np.argsort(sample_ids_fl)

    fl_mu_hat = fl_results["mu_hat"][perm_fl]
    pooled_mu_hat = pooled_dds.layers["_mu_hat"][perm]

    # Check that the nans are at the same places
    assert np.all(np.isnan(fl_mu_hat) == np.isnan(pooled_mu_hat))

    # Replace nans with 1.
    fl_mu_hat[np.isnan(fl_mu_hat)] = 1.0
    pooled_mu_hat[np.isnan(pooled_mu_hat)] = 1.0

    assert np.allclose(
        np.sort(fl_results["mu_hat"].flatten()),
        np.sort(pooled_dds.layers["_mu_hat"].flatten()),
        equal_nan=True,
    )

    # Tests that the genewise dispersions are close to the pooled ones, or if not,
    # that the adjusted log likelihood is close or better

    # Compute relative error

    # If any of the relative errors is above 2%, check likelihoods
    perform_dispersions_and_nll_relative_check(
        fl_results["genewise_dispersions"],
        pooled_dds,
        dispersions_param_name="genewise_dispersions",
        rtol=rtol,
        atol=atol,
        nll_rtol=nll_rtol,
        nll_atol=nll_atol,
        tolerated_failed_genes=tolerated_failed_genes,
    )


class GenewiseDispersionsTester(
    UnitTester,
    ComputeGenewiseDispersions,
):
    """A class to implement a unit test for the genewise dispersions.

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

    min_mu : float
        Lower threshold for mean expression values. (default: ``0.5``).

    beta_tol : float
        Tolerance for the beta coefficients. (default: ``1e-8``).

    max_beta : float
        Upper threshold for the beta coefficients. (default: ``30``).

    irls_num_iter : int
        Number of iterations for the IRLS algorithm. (default: ``20``).

    joblib_backend : str
        The backend to use for the IRLS algorithm. (default: ``"loky"``).

    num_jobs : int
        Number of CPUs to use for parallelization. (default: ``8``).

    joblib_verbosity : int
        Verbosity level for joblib. (default: ``3``).

    irls_batch_size : int
        Batch size for the IRLS algorithm. (default: ``100``).

    PQN_c1 : float
        Parameter for the Proximal Newton algorithm. (default: ``1e-4``) (which
        catches the IRLS algorithm). This is a line search parameter for the Armijo
        condition.

    PQN_ftol : float
        Tolerance for the Proximal Newton algorithm. (default: ``1e-7``).

    PQN_num_iters_ls : int
        Number of iterations for the line search in the Proximal Newton algorithm.
        (default: ``20``).

    PQN_num_iters : int
        Number of iterations for the Proximal Newton algorithm. (default: ``100``).

    PQN_min_mu : float
        Lower threshold for the mean expression values in
        the Proximal Newton algorithm.

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

    save_genewise_dispersions_checkpoints
        Save genewise dispersions checkpoints, that is the genewise dispersions,
        the MoM dispersions, and the mu_hat estimates.

    init_local_states
        Initialize the local states, returning the local gram matrix and local features.

    get_local_mom_dispersions_mu_dispersions
        Collect MoM dispersions and mu_hat estimate and pass on.

    concatenate_mu_estimates
        Concatenate initial mu_hat estimates and pass on dispersions.

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
        min_mu: float = 0.5,
        beta_tol: float = 1e-8,
        max_beta: float = 30,
        irls_num_iter: int = 20,
        joblib_backend: str = "loky",
        joblib_verbosity: int = 3,
        irls_batch_size: int = 100,
        PQN_c1: float = 1e-4,
        PQN_ftol: float = 1e-7,
        PQN_num_iters_ls: int = 20,
        PQN_num_iters: int = 100,
        PQN_min_mu: float = 0.0,
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
            min_mu=min_mu,
            beta_tol=beta_tol,
            max_beta=max_beta,
            irls_num_iter=irls_num_iter,
            joblib_backend=joblib_backend,
            joblib_verbosity=joblib_verbosity,
            irls_batch_size=irls_batch_size,
            PQN_c1=PQN_c1,
            PQN_ftol=PQN_ftol,
            PQN_num_iters_ls=PQN_num_iters_ls,
            PQN_num_iters=PQN_num_iters,
            PQN_min_mu=PQN_min_mu,
        )

        #### Define hyper parameters ####

        self.min_disp = min_disp
        self.max_disp = max_disp
        self.grid_batch_size = grid_batch_size
        self.grid_depth = grid_depth
        self.grid_length = grid_length
        self.num_jobs = num_jobs
        self.min_mu = min_mu
        self.beta_tol = beta_tol
        self.max_beta = max_beta

        # Parameters of the IRLS algorithm
        self.irls_num_iter = irls_num_iter
        self.PQN_c1 = PQN_c1
        self.PQN_ftol = PQN_ftol
        self.PQN_num_iters_ls = PQN_num_iters_ls
        self.PQN_num_iters = PQN_num_iters
        self.PQN_min_mu = PQN_min_mu

        #### Define job parallelization parameters ####
        self.joblib_verbosity = joblib_verbosity
        self.num_jobs = num_jobs
        self.joblib_backend = joblib_backend
        self.irls_batch_size = irls_batch_size

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

        #### Fit genewise dispersions ####
        (
            local_states,
            genewise_dispersions_shared_state,
            round_idx,
        ) = self.fit_genewise_dispersions(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            gram_features_shared_states=shared_states,
            round_idx=round_idx,
            clean_models=clean_models,
        )

        self.save_genewise_dispersions_checkpoints(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            genewise_dispersions_shared_state=genewise_dispersions_shared_state,
            round_idx=round_idx,
            clean_models=False,
        )

    def save_genewise_dispersions_checkpoints(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        genewise_dispersions_shared_state,
        round_idx,
        clean_models,
    ):
        """Save genewise dispersions checkpoints.

        This method saves the genewise dispersions, the MoM dispersions,
        and the mu_hat estimates.

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
        # ---- Get MoM dispersions and mu_hat estimates ---- #

        local_states, shared_states, round_idx = local_step(
            local_method=self.get_local_mom_dispersions_mu_dispersions,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=genewise_dispersions_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Get local MoM dispersions and mu_hat",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        # ---- Concatenate mu_hat estimates ---- #

        results_shared_state, round_idx = aggregation_step(
            aggregation_method=self.concatenate_mu_estimates,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Concatenate mu_hat estimates",
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
        """Initialize the local states.

        Here, we copy the reference dds to the local state, and
        create the local gram matrix and local features, which are necessary inputs
        to the genewise dispersions computation.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener.
        shared_state : Any
            Shared state. Not used.

        Returns
        -------
        dict
            Local states containing a "local_gran_matrix" and a "local_features" fields.
            These fields are used to compute the rough dispersions, and are computed in
            the last step of the compute_size_factors step in the main pipe.
        """

        self.local_adata = self.local_reference_dds.copy()
        del self.local_adata.layers["_mu_hat"]
        # This field is not saved in pydeseq2 but used in fedpyseq2
        self.local_adata.uns["n_params"] = self.local_adata.obsm["design_matrix"].shape[
            1
        ]
        design = self.local_adata.obsm["design_matrix"].values

        return {
            "local_gram_matrix": design.T @ design,
            "local_features": design.T @ self.local_adata.layers["normed_counts"],
        }

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def get_local_mom_dispersions_mu_dispersions(
        self, data_from_opener: ad.AnnData, shared_state: dict
    ) -> dict:
        """Collect MoM dispersions and mu_hat estimate and pass on.

        Here, we pass on the genewise dispersions present in the shared state,
        and collect the MoM dispersions and local mu_hat estimates from the
        local adata.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener.
        shared_state : dict
            Shared state, containing the genewise dispersions in the
            "genewise_dispersions" field.

        Returns
        -------
        dict
            Shared state containing the MoM dispersions in the "MoM_dispersions"
            field, the local mu_hat estimates in the "local_mu_hat" field, and the
            genewise dispersions in the "genewise_dispersions" field.
        """

        return {
            "MoM_dispersions": self.local_adata.varm["_MoM_dispersions"],
            "local_mu_hat": self.local_adata.layers["_mu_hat"],
            "genewise_dispersions": shared_state["genewise_dispersions"],
            "sample_ids": self.local_adata.obs_names,
            "design_columns": self.local_adata.obsm["design_matrix"].columns,
        }

    @remote
    @log_remote
    def concatenate_mu_estimates(self, shared_states: list):
        """Concatenate initial mu_hat estimates and pass on dispersions.

        Parameters
        ----------
        shared_states : list
            A list of shared states with a "local_mu_hat" key containing the
            local mu_hat estimates, a "MoM_dispersions" key containing the MoM
            dispersions, and a "genewise_dispersions" key containing the genewise
            dispersions. The MoM dispersions and gene-wise dispersions are passed on
            and are supposed to be the same across all states.

        """
        mu_hat = np.vstack([state["local_mu_hat"] for state in shared_states])
        sample_ids = np.concatenate([state["sample_ids"] for state in shared_states])
        design_columns = shared_states[0]["design_columns"]
        self.results = {
            "mu_hat": mu_hat,
            "MoM_dispersions": shared_states[0]["MoM_dispersions"],
            "genewise_dispersions": shared_states[0]["genewise_dispersions"],
            "sample_ids": sample_ids,
            "design_columns": design_columns,
        }
