import pickle as pkl
from abc import abstractmethod
from pathlib import Path
from typing import Any

import anndata as ad
from fedpydeseq2_datasets.utils import get_ground_truth_dds_name
from pydeseq2.dds import DeseqDataSet
from substrafl import ComputePlanBuilder
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.remote import remote_data

from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote_data
from fedpydeseq2.core.utils.logging import log_save_local_state
from tests.unit_tests.unit_test_helpers.set_local_reference import SetLocalReference


class UnitTester(
    ComputePlanBuilder,
    SetLocalReference,
):
    """A base semi-abstract class to implement unit tests for DESea2 steps.

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

    refit_cooks: bool
        Whether to refit the model with the Cook's distance. (default: ``False``).

    joblib_backend : str
        The joblib backend to use for parallelization. (default: ``"loky"``).

    save_layers_to_disk : bool
        Whether to save the layers to disk. (default: ``False``).


        The log level to use for the substrafl logger. (default: ``logging.DEBUG``).

    """

    def __init__(
        self,
        design_factors: str | list[str],
        ref_levels: dict[str, str] | None = None,
        continuous_factors: list[str] | None = None,
        contrast: list[str] | None = None,
        reference_data_path: str | Path | None = None,
        reference_dds_ref_level: tuple[str, ...] | None = None,
        refit_cooks: bool = False,
        joblib_backend: str = "loky",
        save_layers_to_disk: bool = False,
        *args,
        **kwargs,
    ):
        # Add all arguments to super init so that they can be retrieved by nodes.
        super().__init__(
            design_factors=design_factors,
            ref_levels=ref_levels,
            continuous_factors=continuous_factors,
            contrast=contrast,
            reference_data_path=reference_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
            refit_cooks=refit_cooks,
            joblib_backend=joblib_backend,
            save_layers_to_disk=save_layers_to_disk,
        )

        #### Define quantities to set the design ####

        # Convert design_factors to list if a single string was provided.
        self.design_factors = (
            [design_factors] if isinstance(design_factors, str) else design_factors
        )

        self.ref_levels = ref_levels
        self.continuous_factors = continuous_factors

        if self.continuous_factors is not None:
            self.categorical_factors = [
                factor
                for factor in self.design_factors
                if factor not in self.continuous_factors
            ]
        else:
            self.categorical_factors = self.design_factors

        self.contrast = contrast

        #### Set attributes to be registered / saved later on ####
        self.results: dict = {}
        self.local_adata: ad.AnnData = None
        self.refit_adata: ad.AnnData = None

        #### Used only if we want the reference
        self.reference_data_path = reference_data_path
        self.reference_dds_name = get_ground_truth_dds_name(
            reference_dds_ref_level, refit_cooks=refit_cooks
        )
        self.local_reference_dds: DeseqDataSet | None = None

        #### Joblib parameters ####
        self.joblib_backend = joblib_backend

        #### Layers paramters
        self.layers_to_load: dict[str, list[str] | None] | None = None
        self.layers_to_save_on_disk: dict[str, list[str] | None] | None = None

        #### Save layers to disk
        self.save_layers_to_disk = save_layers_to_disk

    @abstractmethod
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

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def init_local_states(
        self, data_from_opener: ad.AnnData, shared_state: Any
    ) -> None:
        """Copy the reference dds to the local state.

        If necessary, to overwrite in child classes to add relevant local states.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.
        shared_state : Any
            Shared state. Not used.
        """

        self.local_adata = self.local_reference_dds.copy()
        # Delete the unused "_mu_hat" layer as it will raise errors
        del self.local_adata.layers["_mu_hat"]
        # This field is not saved in pydeseq2 but used in fedpyseq2
        self.local_adata.uns["n_params"] = self.local_adata.obsm["design_matrix"].shape[
            1
        ]

    @log_save_local_state
    def save_local_state(self, path: Path) -> None:
        """Save the local state of the strategy.

        Parameters
        ----------
        path : Path
            Path to the file where to save the state. Automatically handled by subtrafl.
        """
        state_to_save = {
            "results": self.results,
            "local_adata": self.local_adata,
            "refit_adata": self.refit_adata,
            "local_reference_dds": self.local_reference_dds,
        }
        with open(path, "wb") as file:
            pkl.dump(state_to_save, file)

    def load_local_state(self, path: Path) -> Any:
        """Load the local state of the strategy.

        Parameters
        ----------
        path : Path
            Path to the file where to load the state from. Automatically handled by
            subtrafl.
        """
        with open(path, "rb") as file:
            state_to_load = pkl.load(file)

        self.results = state_to_load["results"]
        self.local_adata = state_to_load["local_adata"]
        self.refit_adata = state_to_load["refit_adata"]
        self.local_reference_dds = state_to_load["local_reference_dds"]
        return self

    @property
    def num_round(self):
        """Return the number of round in the strategy.

        TODO do something clever with this.

        Returns
        -------
        int
            Number of round in the strategy.
        """
        return None

    def get_result(self):
        """Return the statistic computed.

        Returns
        -------
        dict
            The global statistics.
        """
        return self.results
