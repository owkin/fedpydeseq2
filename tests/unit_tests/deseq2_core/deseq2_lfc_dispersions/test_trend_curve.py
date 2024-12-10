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
from substrafl.remote import remote_data

from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_dispersion_prior import (  # noqa: E501
    ComputeDispersionPrior,
)
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote_data
from fedpydeseq2.core.utils.pass_on_results import AggPassOnResults
from tests.tcga_testing_pipe import run_tcga_testing_pipe
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
        (["stage", "gender"], None),  # TODO this case fails to converge in PyDESeq2
        (["stage", "gender", "CPE"], ["CPE"]),
    ],
)
def test_trend_curve(
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
    design_factors,
    continuous_factors,
):
    """Perform a unit test for the trend curve.

    Starting with the same genewise dispersions as the reference DeseqDataSet, fit a
    parametric trend curve, compute the prior dispersion and compare the results with
    the reference.

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
    """

    trend_curve_testing_pipe(
        raw_data_path,
        local_processed_data_path,
        tcga_assets_directory,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
    )


def trend_curve_testing_pipe(
    data_path,
    processed_data_path,
    assets_directory,
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
    """Perform a unit test for the trend curve.

    Starting with the same genewise dispersions as the reference DeseqDataSet, fit a
    parametric trend curve, compute the prior dispersion and compare the results with
    the reference.

    Parameters
    ----------
    data_path: Path
        The path to the root data.

    processed_data_path: Path
        The path to the processed data. The subdirectories will
        be created if needed

    assets_directory: Path
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
    # Get FL results.
    fl_results = run_tcga_testing_pipe(
        TrendCurveTester(
            design_factors=design_factors,
            continuous_factors=continuous_factors,
            ref_levels=ref_levels,
            reference_data_path=reference_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
        ),
        raw_data_path=data_path,
        processed_data_path=processed_data_path,
        assets_directory=assets_directory,
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

    if pooled_dds.uns["disp_function_type"] == "mean":
        assert fl_results["disp_function_type"] == "mean"
        np.allclose(
            fl_results["mean_disp"],
            pooled_dds.uns["mean_disp"],
            equal_nan=False,
            rtol=0.02,
        )

        return

    assert np.allclose(
        fl_results["trend_coeffs"],
        pooled_dds.uns["trend_coeffs"],
        equal_nan=False,
        rtol=0.02,
    )

    # Test the dispersion prior
    assert np.allclose(
        fl_results["prior_disp_var"],
        pooled_dds.uns["prior_disp_var"],
        equal_nan=True,
        rtol=0.02,
    )


class TrendCurveTester(
    UnitTester, ComputeDispersionPrior, AggPassOnFirstSharedState, AggPassOnResults
):
    """A class to implement a unit test for the trend curve.

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
        Build the computation graph to compute the trend curve.

    init_local_states
        A remote_data method.
        Copy the reference dds to the local state and add the non_zero mask.
        Return the genewise dispersions as the shared state.
    """

    def __init__(
        self,
        design_factors: str | list[str],
        ref_levels: dict[str, str] | None = None,
        continuous_factors: list[str] | None = None,
        min_disp: float = 1e-8,
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
            reference_data_path=reference_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
        )

        #### Define hyper parameters ####
        self.min_disp = min_disp

    def build_compute_plan(
        self,
        train_data_nodes: list[TrainDataNode],
        aggregation_node: AggregationNode,
        evaluation_strategy=None,
        num_rounds=None,
        clean_models=True,
    ):
        """Build the computation graph to compute the trend curve.

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

        #### Pass the  first shared state as the genewise dispersions shared states ####

        genewise_dispersions_shared_state, round_idx = aggregation_step(
            aggregation_method=self.pass_on_shared_state,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            round_idx=round_idx,
            description="Pass the genewise dispersions shared state",
            clean_models=clean_models,
        )

        #### Fit dispersion trends ####
        (
            local_states,
            dispersion_trend_shared_state,
            round_idx,
        ) = self.compute_dispersion_prior(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            genewise_dispersions_shared_state=genewise_dispersions_shared_state,
            round_idx=round_idx,
            clean_models=clean_models,
        )

        #### Save shared state ####

        aggregation_step(
            aggregation_method=self.pass_on_results,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=[dispersion_trend_shared_state],
            round_idx=round_idx,
            description="Save the first shared state",
            clean_models=False,
        )

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def init_local_states(
        self, data_from_opener: ad.AnnData, shared_state: Any
    ) -> dict:
        """Copy the reference dds to the local state and add the non_zero mask.

        Returns a shared state with the genewise dispersions.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.
        shared_state : Any
            Shared state with a "genewise_dispersions" key.

        Returns
        -------
        dict
            A dictionary containing the genewise dispersions.

        """

        self.local_adata = self.local_reference_dds.copy()
        # Delete the unused "_mu_hat" layer as it will raise errors
        del self.local_adata.layers["_mu_hat"]
        self.local_adata.varm["non_zero"] = self.local_reference_dds.varm["non_zero"]
        self.local_adata.uns["n_params"] = self.local_adata.obsm["design_matrix"].shape[
            -1
        ]

        return {"genewise_dispersions": self.local_adata.varm["genewise_dispersions"]}