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

from fedpydeseq2.core.deseq2_core.compute_size_factors import ComputeSizeFactors
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data
from tests.tcga_testing_pipe import run_tcga_testing_pipe
from tests.unit_tests.unit_test_helpers.unit_tester import UnitTester


@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_size_factors(
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
    ref_levels: dict[str, str] | None = {"stage": "Advanced"},  # noqa: B006
    reference_dds_ref_level: tuple[str, ...] | None = None,
):
    """Perform a unit test for the size factors.

    Starting with the same counts as the reference dataset, compute size factors and
    compare the results with the reference.

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

    ref_levels: dict or None
        The reference levels of the design factors.

    reference_dds_ref_level: tuple or None
        The reference level of the design factors.
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

    reference_data_path = processed_data_path / "centers_data" / "tcga" / experiment_id
    # Get FL results.
    fl_results = run_tcga_testing_pipe(
        SizeFactorsTester(
            design_factors=design_factors,
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

    # TODO size factors are sorted because we don't have access to indexes
    # There could be issues in case of non-unique values
    assert np.allclose(
        np.sort(fl_results["size_factors"]),
        np.sort(pooled_dds.obsm["size_factors"]),
        equal_nan=True,
    )


class SizeFactorsTester(
    UnitTester,
    ComputeSizeFactors,
):
    """A class to implement a unit test for the size factors.

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

    Methods
    -------
    init_local_states
        Local method to initialize the local states.
        It returns the design columns of the local design matrix, which is required to
        start the computation.

    merge_design_columns_and_build_contrast
        Aggregation method to merge the columns of the design matrices and build the
        contrast.
        This method returns a shared state containing
        - merged_columns: the names of the columns that the local design matrices should
            have.
        - contrast: the contrast (in a list of strings form) to be used for the DESeq2
        These are required to start the first local step of the computation
        of size factors.

    compute_local_size_factors
        Local method to compute the size factors.
        Indeed, the compute_size_factors method only returns an aggregated state
        which allows to compute the local size factor and not the size factor itself.
        In the main pipeline, the explicit computation of the local size factors
        is done in the first local step of compute_mom_dispersions.


    concatenate_size_factors
        Aggregation method to concatenate the size factors.
        This method returns the concatenated (pooled equivalent) size factors

    save_size_factors
        Local method to save the size factors, combining the two previous methods
        as well as the build_share_results_tasks method.
    """

    def __init__(
        self,
        design_factors: str | list[str],
        ref_levels: dict[str, str] | None = None,
        continuous_factors: list[str] | None = None,
        reference_data_path: str | Path | None = None,
        reference_dds_ref_level: tuple[str, ...] | None = None,
        contrast: list[str] | None = None,
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
            contrast=contrast,
        )

        self.contrast = contrast

    def build_compute_plan(
        self,
        train_data_nodes: list[TrainDataNode],
        aggregation_node: AggregationNode,
        evaluation_strategy=None,
        num_rounds=None,
        clean_models=True,
    ):
        """Build the computation graph to test computing the size factors.

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

        #### Compute size factors ####
        (
            local_states,
            shared_states,
            round_idx,
        ) = self.compute_size_factors(
            train_data_nodes,
            aggregation_node,
            local_states,
            shared_states=shared_states,
            round_idx=round_idx,
            clean_models=clean_models,
        )

        ### Save results ###
        self.save_size_factors(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=False,
        )

    def save_size_factors(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        round_idx,
        clean_models,
    ):
        """Check size factors.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: dict
            Local states. Required to propagate intermediate results.

        round_idx: int
            The current round index.

        clean_models: bool
            Whether to clean the models after the computation.

        Returns
        -------
        local_states: dict
            Local states. Required to propagate intermediate results.

        round_idx: int
            The updated round index.
        """

        # ---- Compute and share local size factors ----#

        local_states, shared_states, round_idx = local_step(
            local_method=self.get_local_size_factors,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            round_idx=round_idx,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Get the local size factors from the adata",
            clean_models=clean_models,
        )

        # ---- Concatenate local size factors ----#

        aggregation_step(
            aggregation_method=self.concatenate_size_factors,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            round_idx=round_idx,
            description="Concatenating local size factors",
            clean_models=clean_models,
        )

    @remote
    @log_remote
    def concatenate_size_factors(self, shared_states):
        """Concatenate size factors together for registration.

        Use for testing purposes only.

        Parameters
        ----------
        shared_states : list
            List of results (size_factors) from training nodes.

        Returns
        -------
        dict
            Concatenated (pooled) size factors.
        """
        tot_sf = np.hstack(
            [state["size_factors"] for state in shared_states],
        )
        self.results = {"size_factors": tot_sf}

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def init_local_states(
        self, data_from_opener: ad.AnnData, shared_state: Any
    ) -> dict:
        """Copy the reference dds to the local state and compute local log mean.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener, used to compute the local log mean.

        shared_state : None, optional
            Not used.

        Returns
        -------
        dict
            Local mean of logs and number of samples.
        """

        self.local_adata = self.local_reference_dds.copy()
        # Delete the unused "_mu_hat" layer as it will raise errors
        del self.local_adata.layers["_mu_hat"]
        # This field is not saved in pydeseq2 but used in fedpyseq2
        self.local_adata.uns["n_params"] = self.local_adata.obsm["design_matrix"].shape[
            1
        ]

        with np.errstate(divide="ignore"):  # ignore division by zero warnings
            return {
                "log_mean": np.log(data_from_opener.X).mean(axis=0),
                "n_samples": data_from_opener.n_obs,
            }

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def get_local_size_factors(
        self, data_from_opener: ad.AnnData, shared_state: Any
    ) -> dict:
        """Get the local size factors.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            Not used.

        shared_state : None
            Not used.

        Returns
        -------
        dict
            A dictionary containing the size factors in the "size_factors" key.
        """

        return {"size_factors": self.local_adata.obsm["size_factors"]}
