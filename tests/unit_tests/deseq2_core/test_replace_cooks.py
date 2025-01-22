import pickle as pkl
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
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
from fedpydeseq2.core.deseq2_core.replace_outliers import ReplaceCooksOutliers
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
def test_replace_cooks_on_small_genes(
    design_factors,
    continuous_factors,
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
):
    replace_cooks_testing_pipe(
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
        (["stage", "gender"], None),
        (["stage", "gender", "CPE"], ["CPE"]),
    ],
)
@pytest.mark.self_hosted_fast
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
def test_replace_cooks_on_small_samples_on_self_hosted_fast(
    design_factors,
    continuous_factors,
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
):
    replace_cooks_testing_pipe(
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
        (["stage", "gender"], None),
        (["stage", "gender", "CPE"], ["CPE"]),
    ],
)
@pytest.mark.self_hosted_slow
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
def test_replace_cooks_on_self_hosted_slow(
    design_factors,
    continuous_factors,
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
):
    replace_cooks_testing_pipe(
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


def replace_cooks_testing_pipe(
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
        ReplaceOutliersTester(
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

    pooled_dds._replace_outliers()

    if hasattr(pooled_dds, "counts_to_refit"):
        pooled_imputed_counts = pd.DataFrame(
            pooled_dds.counts_to_refit.X,
            index=pooled_dds.counts_to_refit.obs_names,
            columns=pooled_dds.counts_to_refit.var_names,
        )
    else:
        pooled_imputed_counts = pd.DataFrame(index=pooled_dds.obs_names, columns=[])

    # FL counts should be the restriction of the pooled counts to the
    # refitted genes and samples (the fl results are restricted to
    # genes that must be refitted and not only the replaced genes)
    fl_imputed_counts = fl_results["imputed_counts"]
    pooled_imputed_counts = pooled_imputed_counts.loc[
        fl_imputed_counts.index, fl_imputed_counts.columns
    ]

    pd.testing.assert_frame_equal(pooled_imputed_counts, fl_imputed_counts)


class ReplaceOutliersTester(
    UnitTester,
    ReplaceCooksOutliers,
    ComputeCookDistances,
    GetNumReplicates,
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

    min_mu : float
        The minimum value of mu used for Cooks distance computation.

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
            min_replicates=min_replicates,
            trimmed_mean_num_iter=trimmed_mean_num_iter,
        )

        self.min_replicates = min_replicates
        self.min_mu = min_mu
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
            aggregation_method=self.sum_num_samples,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=init_shared_states,
            description="Get the total number of samples.",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        local_states, max_disp_shared_states, round_idx = local_step(
            local_method=self.set_tot_num_samples,
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

        # Compute Cooks distances and impute outliers
        local_states, shared_state, round_idx = self.compute_cook_distance(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models,
        )

        local_states, refit_features_shared_states, round_idx = self.replace_outliers(
            train_data_nodes,
            aggregation_node,
            local_states,
            cooks_shared_state=shared_state,
            round_idx=round_idx,
            clean_models=clean_models,
        )

        ##### Save results #####

        local_states, shared_states, round_idx = local_step(
            local_method=self.loc_return_imputed_counts,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Return imputed counts in a shared state",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        aggregation_step(
            aggregation_method=self.agg_merge_imputed_counts,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Merge the lists of local imputed counts",
            round_idx=round_idx,
            clean_models=False,
        )

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def loc_return_imputed_counts(
        self,
        data_from_opener,
        shared_state: dict,
    ) -> dict:
        """Return the imputed counts as a DataFrame in a shared state.

        Used for testing only.
        """

        return {
            "imputed_counts": pd.DataFrame(
                self.refit_adata.X,
                index=self.refit_adata.obs_names,
                columns=self.refit_adata.var_names,
            )
        }

    @remote
    @log_remote
    def agg_merge_imputed_counts(
        self,
        shared_states: dict,
    ):
        """Merge the imputed counts.

        Used for testing only.
        """

        imputed_counts = pd.concat(
            [shared_state["imputed_counts"] for shared_state in shared_states],
            axis=0,
        )
        self.results = {"imputed_counts": imputed_counts}

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
        # Delete the unused "_mu_hat" layer as it will raise errors
        del self.local_adata.layers["_mu_hat"]
        self.local_adata.varm["non_zero"] = self.local_reference_dds.varm["non_zero"]
        self.local_adata.uns["n_params"] = self.local_reference_dds.obsm[
            "design_matrix"
        ].shape[1]

        return {"num_samples": self.local_adata.n_obs}

    @remote
    @log_remote
    def sum_num_samples(self, shared_states):
        """Compute the total number of samples to set max_disp.

        Parameters
        ----------
        shared_states : List
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
    def set_tot_num_samples(self, data_from_opener: ad.AnnData, shared_state: Any):
        """Set the total number of samples in the study.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : Any
            Shared state with "fitted_dispersions" and "prior_disp_var" keys.
        """

        self.local_adata.uns["tot_num_samples"] = shared_state["tot_num_samples"]
