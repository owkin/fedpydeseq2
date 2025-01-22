"""A class to implement a unit tester class for ComputeLFC."""
from pathlib import Path
from typing import Any
from typing import Literal

import anndata as ad
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_lfc import ComputeLFC
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data
from tests.unit_tests.deseq2_core.deseq2_lfc_dispersions.compute_lfc.substeps import (
    AggConcatenateHandMu,
)
from tests.unit_tests.deseq2_core.deseq2_lfc_dispersions.compute_lfc.substeps import (
    LocGetLocalComputeLFCResults,
)
from tests.unit_tests.unit_test_helpers.unit_tester import UnitTester


class ComputeLFCTester(
    UnitTester,
    ComputeLFC,
    LocGetLocalComputeLFCResults,
    AggConcatenateHandMu,
):
    """A class to implement a unit test for ComputeLFC.

    Parameters
    ----------
    design_factors : str or list
        Name of the columns of metadata to be used as design variables.
        If you are using categorical and continuous factors, you must put
        all of them here.

    lfc_mode : Literal["lfc", "mu_init"]
        The mode of the IRLS algorithm. (default: ``"lfc"``).

    ref_levels : dict or None
        An optional dictionary of the form ``{"factor": "test_level"}``
        specifying for each factor the reference (control) level against which
        we're testing, e.g. ``{"condition", "A"}``. Factors that are left out
        will be assigned random reference levels. (default: ``None``).

    continuous_factors : list or None
        An optional list of continuous (as opposed to categorical) factors. Any factor
        not in ``continuous_factors`` will be considered categorical
        (default: ``None``).

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
    init_local_states
        A remote_data method which sets the local adata and the local gram matrix
        from the reference_dds.

    compute_gram_matrix
        A remote method which computes the gram matrix.

    set_gram_matrix
        A remote_data method which sets the gram matrix in the local adata.

    build_compute_plan
        Build the computation graph to run a ComputeLFC algorithm.

    save_irls_results
        The method to save the IRLS results.
    """

    def __init__(
        self,
        design_factors: str | list[str],
        lfc_mode: Literal["lfc", "mu_init"] = "lfc",
        ref_levels: dict[str, str] | None = None,
        continuous_factors: list[str] | None = None,
        min_mu: float = 0.5,
        beta_tol: float = 1e-8,
        max_beta: float = 30,
        irls_num_iter: int = 20,
        joblib_backend: str = "loky",
        num_jobs: int = 8,
        joblib_verbosity: int = 0,
        irls_batch_size: int = 100,
        PQN_c1: float = 1e-4,
        PQN_ftol: float = 1e-7,
        PQN_num_iters_ls: int = 20,
        PQN_num_iters: int = 100,
        PQN_min_mu: float = 0.0,
        reference_data_path: str | Path | None = None,
        reference_dds_ref_level: tuple[str, ...] | None = None,
    ):
        # Add all arguments to super init so that they can be retrieved by nodes.
        super().__init__(
            design_factors=design_factors,
            lfc_mode=lfc_mode,
            ref_levels=ref_levels,
            continuous_factors=continuous_factors,
            reference_data_path=reference_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
            min_mu=min_mu,
            beta_tol=beta_tol,
            max_beta=max_beta,
            irls_num_iter=irls_num_iter,
            joblib_backend=joblib_backend,
            num_jobs=num_jobs,
            joblib_verbosity=joblib_verbosity,
            irls_batch_size=irls_batch_size,
            PQN_c1=PQN_c1,
            PQN_ftol=PQN_ftol,
            PQN_num_iters_ls=PQN_num_iters_ls,
            PQN_num_iters=PQN_num_iters,
            PQN_min_mu=PQN_min_mu,
        )

        #### Define hyper parameters ####

        self.min_mu = min_mu
        self.beta_tol = beta_tol
        self.max_beta = max_beta

        # Parameters of the IRLS algorithm
        self.lfc_mode = lfc_mode
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

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def init_local_states(
        self, data_from_opener: ad.AnnData, shared_state: Any
    ) -> dict:
        """Initialize the local states.

        This methods sets the local_adata and the local gram matrix.
        from the reference_dds.


        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : Any
            Not used.

        Returns
        -------
        local_states : dict
            Local states containing the local Gram matrix.
        """

        # Using super().init_local_states_from_opener(data_from_opener, shared_state)
        # does not work, so we have to duplicate the code
        self.local_adata = self.local_reference_dds.copy()
        # Delete the unused "_mu_hat" layer
        del self.local_adata.layers["_mu_hat"]
        # This field is not saved in pydeseq2 but used in fedpyseq2
        self.local_adata.uns["n_params"] = self.local_adata.obsm["design_matrix"].shape[
            1
        ]
        # Get the local gram matrix for all genes
        design_matrix = self.local_adata.obsm["design_matrix"].values

        return {
            "local_gram_matrix": design_matrix.T @ design_matrix,
        }

    @remote
    @log_remote
    def compute_gram_matrix(self, shared_states: list[dict]) -> dict:
        """Compute the gram matrix.

        Parameters
        ----------
        shared_states : list
            List of shared states.

        Returns
        -------
        dict
            Dictionary containing the global gram matrix.
        """

        # Sum the local gram matrices
        tot_gram_matrix = sum([state["local_gram_matrix"] for state in shared_states])
        # Share it with the centers
        return {
            "global_gram_matrix": tot_gram_matrix,
        }

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def set_gram_matrix(self, data_from_opener: ad.AnnData, shared_state: Any):
        """Set the gram matrix in the local adata.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : Any
            Shared state containing the global gram matrix.
        """
        self.local_adata.uns["_global_gram_matrix"] = shared_state["global_gram_matrix"]

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

        #### Compute the gram matrix ####

        gram_matrix_shared_state, round_idx = aggregation_step(
            aggregation_method=self.compute_gram_matrix,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            round_idx=round_idx,
            description="Compute the global gram matrix.",
            clean_models=clean_models,
        )

        local_states, shared_states, round_idx = local_step(
            local_method=self.set_gram_matrix,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=gram_matrix_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Set the global gram matrix in the local adata.",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        #### Perform fedIRLS ####

        local_states, round_idx = self.compute_lfc(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=clean_models,
            lfc_mode=self.lfc_mode,
        )

        self.save_irls_results(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=False,
        )

    def save_irls_results(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        round_idx,
        clean_models,
    ):
        """Save the IRLS results.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: dict
            Local states. Required to propagate intermediate results.

        round_idx: int
            The current round.

        clean_models: bool
            If True, the models are cleaned.
        """
        (
            local_states,
            local_irls_results_shared_states,
            round_idx,
        ) = local_step(
            local_method=self.get_local_irls_results,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            round_idx=round_idx,
            input_local_states=local_states,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Get the local IRLS results.",
            clean_models=clean_models,
        )

        aggregation_step(
            aggregation_method=self.concatenate_irls_outputs,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=local_irls_results_shared_states,
            round_idx=round_idx,
            description="Compute global IRLS inverse hat matrix and last nll.",
            clean_models=clean_models,
        )
