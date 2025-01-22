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

from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions import DESeq2LFCDispersions
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions.get_num_replicates import (  # noqa: E501
    GetNumReplicates,
)
from fedpydeseq2.core.deseq2_core.replace_outliers import ReplaceCooksOutliers
from fedpydeseq2.core.deseq2_core.replace_refitted_values import ReplaceRefittedValues
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
        (["stage", "gender", "CPE"], ["CPE"]),
    ],
)
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_refit_cooks_on_small_genes(
    design_factors,
    continuous_factors,
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
):
    refit_cooks_testing_pipe(
        raw_data_path,
        processed_data_path=processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        small_samples=False,
        small_genes=True,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
        design_factors="stage",
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
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_refit_cooks_on_small_samples_on_self_hosted_fast(
    design_factors,
    continuous_factors,
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
):
    refit_cooks_testing_pipe(
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
    "design_factors, continuous_factors",
    [
        ("stage", None),
        (["stage", "gender", "CPE"], ["CPE"]),
    ],
)
@pytest.mark.self_hosted_slow
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_refit_cooks_on_self_hosted_slow(
    design_factors,
    continuous_factors,
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
):
    refit_cooks_testing_pipe(
        raw_data_path,
        processed_data_path=processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
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


def refit_cooks_testing_pipe(
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
    """Perform a unit test for outlier imputation.

    Starting with the same counts as the reference dataset, replace count values of
    Cooks outliers and compare with the reference.

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
        An optional list of continuous (as opposed to categorical) factors. Any factor
        not in ``continuous_factors`` will be considered categorical.

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
        RefitOutliersTester(
            design_factors=design_factors,
            continuous_factors=continuous_factors,
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

    fl_adata = fl_results["local_adatas"][0]

    min_disp = fl_results["min_disp"]
    max_disp = fl_adata.uns["max_disp"]
    max_beta = fl_results["max_beta"]

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

    init_pooled_dds = pooled_dds.copy()
    pooled_dds.refit_cooks = True
    pooled_dds.refit()

    # Check that the refitted size factors are the same as the original values
    for adata in fl_results["local_adatas"]:
        assert np.allclose(
            adata.obsm["size_factors"],
            init_pooled_dds[adata.obs_names].obsm["size_factors"],
            equal_nan=True,
        )

    # Check that refitted genes are the same
    assert np.array_equal(
        fl_adata.varm["refitted"], pooled_dds.varm["refitted"], equal_nan=True
    )

    # Check that replaceable samples are the same
    for adata in fl_results["local_adatas"]:
        assert np.array_equal(
            adata.obsm["replaceable"],
            pooled_dds[adata.obs_names].obsm["replaceable"],
            equal_nan=True,
        )

    if pooled_dds.varm["refitted"].sum() > 0:
        # Check that the genewise dispersions of refitted genes have changed
        # Except if out of bounds
        fl_genewise_on_refitted = fl_adata.varm["genewise_dispersions"][
            fl_adata.varm["refitted"]
        ]
        init_pooled_genewise_on_refitted = init_pooled_dds.varm["genewise_dispersions"][
            pooled_dds.varm["refitted"]
        ]
        different = fl_genewise_on_refitted != init_pooled_genewise_on_refitted
        out_of_bounds = (fl_genewise_on_refitted < min_disp + 1e-8) | (
            fl_genewise_on_refitted > max_disp - 1e-8
        )
        if out_of_bounds.sum() > 0:
            print("Genewise dispersions out of bounds")
            print(fl_adata.var_names[fl_adata.varm["refitted"]][out_of_bounds])
            print(fl_genewise_on_refitted[out_of_bounds])
            print(init_pooled_genewise_on_refitted[out_of_bounds])
        assert np.all(different | out_of_bounds)

        # Check that the MAP dispersions of refitted genes have changed
        # Except if out of bounds
        fl_MAP_on_refitted = fl_adata.varm["MAP_dispersions"][fl_adata.varm["refitted"]]
        init_pooled_MAP_on_refitted = init_pooled_dds.varm["MAP_dispersions"][
            pooled_dds.varm["refitted"]
        ]
        different = fl_MAP_on_refitted != init_pooled_MAP_on_refitted
        out_of_bounds = (fl_MAP_on_refitted < min_disp + 1e-8) | (
            fl_MAP_on_refitted > max_disp - 1e-8
        )
        if out_of_bounds.sum() > 0:
            print("MAP dispersions out of bounds")
            print(fl_adata.var_names[fl_adata.varm["refitted"]][out_of_bounds])
            print(fl_MAP_on_refitted[out_of_bounds])
            print(init_pooled_MAP_on_refitted[out_of_bounds])
        assert np.all(different | out_of_bounds)

        # Check that the LFC of refitted genes have changed
        # Except if out of bounds
        fl_LFC_on_refitted = fl_adata.varm["LFC"][fl_adata.varm["refitted"]].to_numpy()
        init_pooled_LFC_on_refitted = init_pooled_dds.varm["LFC"][
            pooled_dds.varm["refitted"]
        ].to_numpy()
        different = fl_LFC_on_refitted != init_pooled_LFC_on_refitted
        out_of_bounds = (fl_LFC_on_refitted < -max_beta + 1e-8) | (
            fl_LFC_on_refitted > max_beta - 1e-8
        )
        if out_of_bounds.sum() > 0:
            print("LFC out of bounds")
            print(fl_adata.var_names[fl_adata.varm["refitted"]][out_of_bounds])
            print(fl_LFC_on_refitted[out_of_bounds])
            print(init_pooled_LFC_on_refitted[out_of_bounds])
        assert np.all(different | out_of_bounds)

    if (~pooled_dds.varm["refitted"]).sum() > 0:
        # Check that the genewise dispersions of non-refitted genes have not changed
        np.testing.assert_array_almost_equal(
            fl_adata.varm["genewise_dispersions"][~fl_adata.varm["refitted"]],
            pooled_dds.varm["genewise_dispersions"][~pooled_dds.varm["refitted"]],
            decimal=6,
        )

        # Check that the MAP dispersions of non-refitted genes have not changed
        np.testing.assert_array_almost_equal(
            fl_adata.varm["MAP_dispersions"][~fl_adata.varm["refitted"]],
            pooled_dds.varm["MAP_dispersions"][~pooled_dds.varm["refitted"]],
            decimal=6,
        )

        # Check that the LFC of non-refitted genes have not changed
        np.testing.assert_array_almost_equal(
            fl_adata.varm["LFC"][~fl_adata.varm["refitted"]].values,
            pooled_dds.varm["LFC"][~pooled_dds.varm["refitted"]].values,
            decimal=6,
        )


class RefitOutliersTester(
    UnitTester,
    ReplaceCooksOutliers,
    DESeq2LFCDispersions,
    GetNumReplicates,
    ReplaceRefittedValues,
):
    """A class to implement a unit test for outlier imputation.

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

    min_replicates : int
        The minimum number of replicates for a gene to be considered for outlier
        replacement. (default: 7).

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
        min_replicates: int = 7,
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
        joblib_verbosity: int = 0,
        irls_batch_size: int = 100,
        independent_filter: bool = True,
        alpha: float = 0.05,
        PQN_c1: float = 1e-4,
        PQN_ftol: float = 1e-7,
        PQN_num_iters_ls: int = 20,
        PQN_num_iters: int = 100,
        PQN_min_mu: float = 0.0,
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
            min_replicates=min_replicates,
            min_disp=min_disp,
            max_disp=max_disp,
            grid_batch_size=grid_batch_size,
            grid_depth=grid_depth,
            grid_length=grid_length,
            num_jobs=num_jobs,
            min_mu=min_mu,
            beta_tol=beta_tol,
            max_beta=max_beta,
            irls_num_iter=irls_num_iter,
            joblib_backend=joblib_backend,
            joblib_verbosity=joblib_verbosity,
            irls_batch_size=irls_batch_size,
            independent_filter=independent_filter,
            alpha=alpha,
            PQN_c1=PQN_c1,
            PQN_ftol=PQN_ftol,
            PQN_num_iters_ls=PQN_num_iters_ls,
            PQN_num_iters=PQN_num_iters,
            PQN_min_mu=PQN_min_mu,
            trimmed_mean_num_iter=trimmed_mean_num_iter,
        )

        #### Define hyper parameters ####

        self.min_disp = min_disp
        self.max_disp = max_disp
        self.grid_batch_size = grid_batch_size
        self.grid_depth = grid_depth
        self.grid_length = grid_length
        self.min_mu = min_mu
        self.beta_tol = beta_tol
        self.max_beta = max_beta

        # Parameters of the IRLS algorithm
        self.irls_num_iter = irls_num_iter
        self.min_replicates = min_replicates
        self.PQN_c1 = PQN_c1
        self.PQN_ftol = PQN_ftol
        self.PQN_num_iters_ls = PQN_num_iters_ls
        self.PQN_num_iters = PQN_num_iters
        self.PQN_min_mu = PQN_min_mu

        # Parameters for the trimmed mean
        self.trimmed_mean_num_iter = trimmed_mean_num_iter

        #### Define job parallelization parameters ####
        self.num_jobs = num_jobs
        self.joblib_verbosity = joblib_verbosity
        self.joblib_backend = joblib_backend
        self.irls_batch_size = irls_batch_size

        # Save on disk
        self.layers_to_save_on_disk = {
            "local_adata": ["cooks"],
            "refit_adata": ["cooks"],
        }

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
        train_data_nodes : List[TrainDataNode]
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
            aggregation_method=self.sum_num_samples_and_gram_matrix,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=init_shared_states,
            description="Get the total number of samples.",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        local_states, max_disp_shared_states, round_idx = local_step(
            local_method=self.set_tot_num_samples_and_gram_matrix,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Set the total number of samples locally.",
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

        local_states, gram_features_shared_state, round_idx = self.replace_outliers(
            train_data_nodes,
            aggregation_node,
            local_states,
            cooks_shared_state=None,
            round_idx=round_idx,
            clean_models=clean_models,
        )

        ##### Run the pipelline in refit mode #####

        local_states, round_idx = self.run_deseq2_lfc_dispersions(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            gram_features_shared_states=gram_features_shared_state,
            round_idx=round_idx,
            clean_models=clean_models,
            refit_mode=True,
        )

        # Replace values in the main ``local_adata`` object
        local_states, round_idx = self.replace_refitted_values(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            round_idx=round_idx,
            clean_models=clean_models,
        )

        # Concatenate local_adatas and store results
        # 1 - Local centers return their local adatas in a local state
        local_states, shared_states, round_idx = local_step(
            local_method=self.get_adatas,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Return local and refit adatas in a shared state",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        # 2 - Return concatenated adatas
        aggregation_step(
            aggregation_method=self.return_adatas,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Return the lists of local and refit adatas",
            round_idx=round_idx,
            clean_models=False,
        )

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def init_local_states(
        self, data_from_opener: ad.AnnData, shared_state: Any
    ) -> dict:
        """Copy the reference dds and add the total number of samples.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : Any
            Shared state with a "num_samples" key.
        """

        self.local_adata = self.local_reference_dds.copy()
        self.local_adata.varm["non_zero"] = self.local_reference_dds.varm["non_zero"]
        self.local_adata.uns["n_params"] = self.local_reference_dds.obsm[
            "design_matrix"
        ].shape[1]
        design_matrix = self.local_adata.obsm["design_matrix"].values

        return {
            "num_samples": self.local_adata.n_obs,
            "local_gram_matrix": design_matrix.T @ design_matrix,
        }

    @remote
    @log_remote
    def sum_num_samples_and_gram_matrix(self, shared_states):
        """Compute the total number of samples to set max_disp and gram matrix.

        Parameters
        ----------
        shared_states : List
            List of initial shared states copied from the reference adata.

        Returns
        -------
        dict
        """
        tot_num_samples = np.sum([state["num_samples"] for state in shared_states])
        # Sum the local gram matrices
        tot_gram_matrix = sum([state["local_gram_matrix"] for state in shared_states])
        return {
            "tot_num_samples": tot_num_samples,
            "global_gram_matrix": tot_gram_matrix,
        }

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def set_tot_num_samples_and_gram_matrix(
        self, data_from_opener: ad.AnnData, shared_state: Any
    ):
        """Set the total number of samples in the study, and the gram matrix.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : Any
            Shared state with "fitted_dispersions" and "prior_disp_var" keys.
        """

        self.local_adata.uns["tot_num_samples"] = shared_state["tot_num_samples"]
        self.local_adata.uns["max_disp"] = max(
            self.max_disp, shared_state["tot_num_samples"]
        )
        # TODO this is not used but the key is expected
        self.local_adata.uns["mean_disp"] = None
        self.local_adata.uns["_global_gram_matrix"] = shared_state["global_gram_matrix"]

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def get_adatas(
        self,
        data_from_opener,
        shared_state: dict,
    ) -> dict:
        """Return adatas.

        Used for testing only.
        """

        return {
            "local_adata": self.local_adata,
            "refit_adata": self.refit_adata,
            "min_disp": self.min_disp,
            "max_beta": self.max_beta,
        }

    @remote
    @log_remote
    def return_adatas(
        self,
        shared_states: list,
    ):
        """Return the adatas as lists.

        Used for testing only.
        """

        local_adatas = [shared_state["local_adata"] for shared_state in shared_states]

        refit_adatas = [shared_state["refit_adata"] for shared_state in shared_states]

        self.results = {
            "local_adatas": local_adatas,
            "refit_adatas": refit_adatas,
            "min_disp": shared_states[0]["min_disp"],
            "max_beta": shared_states[0]["max_beta"],
        }
