"""A class to implement a unit tester class for FedIRLS."""
from pathlib import Path

import anndata as ad
import numpy as np
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.fed_algorithms import FedIRLS
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data
from fedpydeseq2.core.utils.pass_on_results import AggPassOnResults
from tests.unit_tests.fed_algorithms.fed_IRLS_PQN_tester import FedIRLSPQNTester
from tests.unit_tests.fed_algorithms.fed_IRLS_PQN_tester import compute_initial_beta
from tests.unit_tests.unit_test_helpers.pass_on_first_shared_state import (
    AggPassOnFirstSharedState,
)


class FedIRLSTester(
    FedIRLSPQNTester, FedIRLS, AggPassOnResults, AggPassOnFirstSharedState
):
    """A class to implement a unit test for FedIRLS.

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
        Build the computation graph to run a FedIRLS algorithm.

    save_irls_results
        The method to save the IRLS results.
    """

    def __init__(
        self,
        design_factors: str | list[str],
        ref_levels: dict[str, str] | None = None,
        continuous_factors: list[str] | None = None,
        min_mu: float = 0.5,
        beta_tol: float = 1e-8,
        max_beta: float = 30,
        irls_num_iter: int = 20,
        joblib_backend: str = "loky",
        num_jobs: int = 8,
        joblib_verbosity: int = 3,
        irls_batch_size: int = 100,
        reference_data_path: str | Path | None = None,
        reference_dds_ref_level: tuple[str, ...] | None = None,
    ):
        # Add all arguments to super init so that they can be retrieved by nodes.
        super().__init__(
            design_factors=design_factors,
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
        )

        #### Define hyper parameters ####

        self.min_mu = min_mu
        self.beta_tol = beta_tol
        self.max_beta = max_beta

        # Parameters of the IRLS algorithm
        self.irls_num_iter = irls_num_iter

        #### Define job parallelization parameters ####
        self.joblib_verbosity = joblib_verbosity
        self.num_jobs = num_jobs
        self.joblib_backend = joblib_backend
        self.irls_batch_size = irls_batch_size

        self.layers_to_save_on_disk = {
            "local_adata": [
                "_mu_hat",
            ],
            "refit_adata": [
                None,
            ],
        }

    @remote
    @log_remote
    def compute_start_state(self, shared_states: list[dict]) -> dict:
        """Compute the beta initialization, and share to the centers.

        Parameters
        ----------
        shared_states : list
            List of shared states.

        Returns
        -------
        dict
            Dictionary containing the starting state of the IRLS algorithm.
            It contains the following keys:
            - beta: ndarray
                The current beta, of shape (n_non_zero_genes, n_params).
            - irls_diverged_mask: ndarray
                A boolean mask indicating if fed avg should be used for a given gene
                (shape: (n_non_zero_genes,)). It is initialized to False.
            - irls_mask: ndarray
                A boolean mask indicating if IRLS should be used for a given gene
                (shape: (n_non_zero_genes,)). It is initialized to True.
            - global_nll: ndarray
                The global_nll of the current beta from the previous beta, of shape
                (n_non_zero_genes,). It is initialized to 1000.0.
            - round_number_irls: int
                The current round number of the IRLS algorithm. It is initialized to 0.
        """

        beta_init = compute_initial_beta(shared_states)
        n_non_zero_genes = beta_init.shape[0]

        # Share it with the centers
        return {
            "beta": beta_init,
            "irls_mask": np.full(n_non_zero_genes, True),
            "irls_diverged_mask": np.full(n_non_zero_genes, False),
            "global_nll": np.full(n_non_zero_genes, 1000.0),
            "round_number_irls": 0,
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

        # This step also shares the counts and design matrix to compute
        # the beta initialization directly

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

        #### Compute the initialization shared state ####

        starting_shared_state, round_idx = aggregation_step(
            aggregation_method=self.compute_start_state,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            round_idx=round_idx,
            description="Compute the global gram matrix.",
            clean_models=clean_models,
        )

        #### Set the beta init in the local states ####

        local_states, shared_states, round_idx = local_step(
            local_method=self.set_beta_init,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=starting_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Set the beta init",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        starting_shared_state, round_idx = aggregation_step(
            aggregation_method=self.pass_on_shared_state,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            round_idx=round_idx,
            description="Pass on the shared state",
            clean_models=clean_models,
        )

        #### Perform fed PQN ####

        local_states, irls_shared_state, round_idx = self.run_fed_irls(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            input_shared_state=starting_shared_state,
            round_idx=round_idx,
            clean_models=clean_models,
        )

        #### Share the results ####

        # Local step to add non_zero genes to the shared state
        local_states, shared_states, round_idx = local_step(
            local_method=self.local_add_non_zero_genes,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=irls_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Add non zero genes to the shared state",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        # Save the results
        aggregation_step(
            aggregation_method=self.pass_on_results,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            round_idx=round_idx,
            description="Save the first shared state",
            clean_models=False,
        )

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def local_add_non_zero_genes(
        self, data_from_opener: ad.AnnData, shared_state: dict
    ) -> dict:
        """Initialize the local states.

        This methods sets the local_adata and the local gram matrix.
        from the reference_dds.


        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict
            Shared state which comes from the last iteration of
            the Fed Prox Quasi Newton algorithm.
            Contains a `beta` and a `PQN_diverged_mask` field.

        Returns
        -------
        local_state : dict
            The local state containing the non zero genes mask and genes
            as an addition to the input shared state.
        """
        non_zero_genes_names = self.local_adata.var_names[
            self.local_adata.varm["non_zero"]
        ]
        non_zero_genes_mask = self.local_adata.varm["non_zero"]
        return {
            "irls_diverged_mask": shared_state["irls_diverged_mask"]
            | shared_state["irls_mask"],
            "beta": shared_state["beta"],
            "non_zero_genes_mask": non_zero_genes_mask,
            "non_zero_genes_names": non_zero_genes_names,
        }
