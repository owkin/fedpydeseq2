from typing import Any
from typing import Literal

import numpy as np
from anndata import AnnData
from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.fed_algorithms.fed_PQN.utils import (
    compute_ascent_direction_decrement,
)
from fedpydeseq2.core.fed_algorithms.fed_PQN.utils import (
    compute_gradient_scaling_matrix_fisher,
)
from fedpydeseq2.core.fed_algorithms.fed_PQN.utils import (
    make_fisher_gradient_nll_step_sizes_batch,
)
from fedpydeseq2.core.utils.compute_lfc_utils import get_lfc_utils_from_gene_mask_adata
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data


class LocMakeFedPQNFisherGradientNLL:
    """Mixin to compute local values, gradient and Fisher information of the NLL.

    Attributes
    ----------
    local_adata : AnnData
        The local AnnData.
    num_jobs : int
        The number of cpus to use.
    joblib_verbosity : int
        The joblib verbosity.
    joblib_backend : str
        The backend to use for the IRLS algorithm.
    irls_batch_size : int
        The batch size to use for the IRLS algorithm.
    max_beta : float
        The maximum value for the beta parameter.
    PQN_num_iters_ls : int
        The number of iterations to use for the line search.
    PQN_min_mu : float
        The min_mu parameter for the Proximal Quasi Newton algorithm.

    Methods
    -------
    make_local_fisher_gradient_nll
        A remote_data method.
        Make the local nll, gradient and fisher matrix.
    """

    local_adata: AnnData
    refit_adata: AnnData
    num_jobs: int
    joblib_verbosity: int
    joblib_backend: str
    irls_batch_size: int
    max_beta: float
    PQN_num_iters_ls: int
    PQN_min_mu: float

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def make_local_fisher_gradient_nll(
        self,
        data_from_opener: AnnData,
        shared_state: dict[str, Any],
        first_iteration_mode: Literal["irls_catch"] | None = None,
        refit_mode: bool = False,
    ):
        r"""Make the local nll, gradient and fisher information matrix.

        Given an ascent direction :math:`d` (an ascent direction being positively
        correlated to the gradient of the starting point) and a starting point
        :math:`beta`, this function
        computes the nll, gradient and Fisher information at the points
        :math:`beta + t * d`,
        for :math:`t` in step_sizes
        (step sizes are :math:`0.5^i` for :math:`i` in :math:`0,...,19`.


        Moreover, if the iteration is the first one, the step sizes are not used,
        and instead, the nll, gradient and fisher information are computed at the
        current beta values.

        Parameters
        ----------
        data_from_opener : AnnData
            Not used.

        shared_state : dict
            A dictionary containing the following
            keys:
            - PQN_mask: ndarray
                A boolean mask indicating if the gene should be used for the
                proximal newton step.
                It is of shape (n_non_zero_genes,)
                Used, but not modified.
            - round_number_PQN: int
                The current round number of the prox newton algorithm.
                Used but not modified.
            - ascent_direction_on_mask: Optional[ndarray]
                The ascent direction, of shape (n_genes, n_params), where
                n_genes is the current number of genes that are active (True
                in the PQN_mask).
                Used but not modified.
            - beta: ndarray
                The log fold changes, of shape (n_non_zero_genes, n_params).
                Used but not modified.
            - global_reg_nll: ndarray
                The global regularized nll, of shape (n_non_zero_genes,).
                Not used and not modified.
            - newton_decrement_on_mask: Optional[ndarray]
                The newton decrement, of shape (n_ngenes,).
                It is None at the first round of the prox newton algorithm.
                Not used and not modified.
            - irls_diverged_mask: ndarray
                A boolean mask indicating if the gene has diverged in the IRLS
                algorithm.
                Not used and not modified.
            - PQN_diverged_mask: ndarray
                A boolean mask indicating if the gene has diverged in the prox newton
                algorithm.
                Not used and not modified.

        first_iteration_mode : Optional[Literal["irls_catch"]]
            For the first iteration, this function behaves differently. If
            first_iteration_mode is None, then we are not at the first iteration.
            If first_iteration_mode is not None, the function will expect a
            different shared state than the one described above, and will construct
            the initial shared state from it.
            If first_iteration_mode is "irls_catch", then we assume that
            we are using the PQN algorithm as a method to catch IRLS when it fails
            The function will expect a
            shared state that contains the following fields:
            - beta: ndarray
                The log fold changes, of shape (n_non_zero_genes, n_params).
            - irls_diverged_mask: ndarray
                A boolean mask indicating if the gene has diverged in the IRLS
                algorithm.
            - irls_mask : ndarray
                The mask of genes that were still active for the IRLS algorithm.

        refit_mode : bool
            Whether to run on `refit_adata`s instead of `local_adata`s.
            (default: False).

        Returns
        -------
        dict
            The state to share to the server.
            It contains the following fields:
            - beta: ndarray
                The log fold changes, of shape (n_non_zero_genes, n_params).
            - local_nll: ndarray
                The local nll, of shape (n_step_sizes, n_genes,), where
                n_genes is the current number of genes that are active (True
                in the PQN_mask). n_step_sizes is the number of step sizes
                considered, which is `PQN_num_iters_ls` if we are not at the
                first round, and 1 otherwise.
                This is created during this step.
            - local_fisher: ndarray
                The local fisher matrix,
                of shape (n_step_sizes, n_genes, n_params, n_params).
                This is created during this step.
            - local_gradient: ndarray
                The local gradient, of shape (n_step_sizes, n_genes, n_params).
                This is created during this step.
            - PQN_diverged_mask: ndarray
                A boolean mask indicating if the gene has diverged in the prox newton
                algorithm.
            - PQN_mask: ndarray
                A boolean mask indicating if the gene should be used for the
                proximal newton step, of shape (n_non_zero_genes,).
            - global_reg_nll: ndarray
                The global regularized nll, of shape (n_non_zero_genes,).
            - newton_decrement_on_mask: Optional[ndarray]
                The newton decrement, of shape (n_ngenes,).
                This is None at the first round of the prox newton algorithm.
            - round_number_PQN: int
                The current round number of the prox newton algorithm.
            - irls_diverged_mask: ndarray
                A boolean mask indicating if the gene has diverged in the IRLS
                algorithm.
            - ascent_direction_on_mask: Optional[ndarray]
                The ascent direction, of shape (n_genes, n_params), where
                n_genes is the current number of genes that are active (True
                in the PQN_mask).

        Raises
        ------
        ValueError
            If first_iteration_mode is not None or "irls_catch".
        """
        if refit_mode:
            adata = self.refit_adata
        else:
            adata = self.local_adata

        # Distinguish between the first iteration and the rest
        if first_iteration_mode is not None and first_iteration_mode == "irls_catch":
            beta = shared_state["beta"]
            irls_diverged_mask = shared_state["irls_diverged_mask"]
            irls_mask = shared_state["irls_mask"]
            PQN_mask = irls_mask | irls_diverged_mask
            irls_diverged_mask = PQN_mask.copy()
            round_number_PQN = 0
            ascent_direction_on_mask = None
            newton_decrement_on_mask = None
            PQN_diverged_mask = np.zeros_like(irls_mask, dtype=bool)
            global_reg_nll = np.nan * np.ones_like(irls_mask, dtype=float)
        elif first_iteration_mode is None:
            # If we are not at the first iteration, we use the shared state
            PQN_mask = shared_state["PQN_mask"]
            round_number_PQN = shared_state["round_number_PQN"]
            ascent_direction_on_mask = shared_state["ascent_direction_on_mask"]
            beta = shared_state["beta"]
            PQN_diverged_mask = shared_state["PQN_diverged_mask"]
            newton_decrement_on_mask = shared_state["newton_decrement_on_mask"]
            global_reg_nll = shared_state["global_reg_nll"]
            irls_diverged_mask = shared_state["irls_diverged_mask"]
        else:
            raise ValueError("first_iteration_mode should be None or 'irls_catch'")

        if round_number_PQN == 0:
            # Sanity check that this is the first round of fed prox
            beta[PQN_mask] = adata.uns["_irls_beta_init"][PQN_mask]
            step_sizes: np.ndarray | None = None

        else:
            step_sizes = 0.5 ** np.arange(self.PQN_num_iters_ls)

        # Get the quantities stored in the adata
        disp_param_name = adata.uns["_irls_disp_param_name"]

        (
            PQN_gene_names,
            design_matrix,
            size_factors,
            counts,
            dispersions,
            beta_on_mask,
        ) = get_lfc_utils_from_gene_mask_adata(
            adata,
            PQN_mask,
            disp_param_name,
            beta=beta,
        )

        # ---- Compute local nll, gradient and Fisher information ---- #

        with parallel_backend(self.joblib_backend):
            res = Parallel(n_jobs=self.num_jobs, verbose=self.joblib_verbosity)(
                delayed(make_fisher_gradient_nll_step_sizes_batch)(
                    design_matrix=design_matrix,
                    size_factors=size_factors,
                    beta=beta_on_mask[i : i + self.irls_batch_size],
                    dispersions=dispersions[i : i + self.irls_batch_size],
                    counts=counts[:, i : i + self.irls_batch_size],
                    ascent_direction=ascent_direction_on_mask[
                        i : i + self.irls_batch_size
                    ]
                    if ascent_direction_on_mask is not None
                    else None,
                    step_sizes=step_sizes,
                    beta_min=-self.max_beta,
                    beta_max=self.max_beta,
                    min_mu=self.PQN_min_mu,
                )
                for i in range(0, len(beta_on_mask), self.irls_batch_size)
            )

        n_step_sizes = len(step_sizes) if step_sizes is not None else 1
        if len(res) == 0:
            H = np.zeros((n_step_sizes, 0, beta.shape[1], beta.shape[1]))
            gradient = np.zeros((n_step_sizes, 0, beta.shape[1]))
            local_nll = np.zeros((n_step_sizes, 0))
        else:
            H = np.concatenate([r[0] for r in res], axis=1)
            gradient = np.concatenate([r[1] for r in res], axis=1)
            local_nll = np.concatenate([r[2] for r in res], axis=1)

        # Create the shared state
        return {
            "beta": beta,
            "local_nll": local_nll,
            "local_fisher": H,
            "local_gradient": gradient,
            "PQN_diverged_mask": PQN_diverged_mask,
            "PQN_mask": PQN_mask,
            "global_reg_nll": global_reg_nll,
            "newton_decrement_on_mask": newton_decrement_on_mask,
            "round_number_PQN": round_number_PQN,
            "irls_diverged_mask": irls_diverged_mask,
            "ascent_direction_on_mask": ascent_direction_on_mask,
        }


class AggChooseStepComputeAscentDirection:
    """Mixin class to compute the right ascent direction.

    An ascent direction is a direction that is positively correlated to the gradient.
    This direction will be used to compute the next iterate in the proximal quasi newton
    algorithm. As our aim will be to mimimize the negative log likelihood, we will
    move in the opposite direction, that is in the direction of minus the
    ascent direction.

    Attributes
    ----------
    num_jobs : int
        The number of cpus to use.
    joblib_verbosity : int
        The joblib verbosity.
    joblib_backend : str
        The backend to use for the IRLS algorithm.
    irls_batch_size : int
        The batch size to use for the IRLS algorithm.
    max_beta : float
        The maximum value for the beta parameter.
    beta_tol : float
        The tolerance for the beta parameter.
    PQN_num_iters_ls : int
        The number of iterations to use for the line search.
    PQN_c1 : float
        The c1 parameter for the line search.
    PQN_ftol : float
        The ftol parameter for the line search.
    PQN_num_iters : int
        The number of iterations to use for the proximal quasi newton algorithm.

    Methods
    -------
    choose_step_and_compute_ascent_direction
        A remote method.
        Choose the best step size and compute the next ascent direction.
    """

    num_jobs: int
    joblib_verbosity: int
    joblib_backend: str
    irls_batch_size: int
    max_beta: float
    beta_tol: float
    PQN_num_iters_ls: int
    PQN_c1: float
    PQN_ftol: float
    PQN_num_iters: int

    @remote
    @log_remote
    def choose_step_and_compute_ascent_direction(
        self, shared_states: list[dict]
    ) -> dict[str, Any]:
        """Choose best step size and compute next ascent direction.

        By "ascent direction", we mean the direction that is positively correlated
        with the gradient.

        The role of this function is twofold.

        1) It chooses the best step size for each gene, and updates the beta values
        as well as the nll values. This allows to define the next iterate.
        Note that at the first iterate, it simply computes the nll, gradient and fisher
        information at the current beta values, to define the next ascent direction.

        2) For this new iterate (or the current one if we are at the first round),
        it computes the gradient scaling matrix, which is used to scale the gradient
        in the proximal newton algorithm. From this gradient scaling matrix, and the
        gradient, it computes the ascent direction (and the newton decrement).


        Parameters
        ----------
        shared_states: list[dict]
            A list of dictionaries containing the following
            keys:
            - beta: ndarray
                The log fold changes, of shape (n_non_zero_genes, n_params).
            - local_nll: ndarray
                The local nll, of shape (n_genes,), where
                n_genes is the current number of genes that are active (True
                in the PQN_mask).
            - local_fisher: ndarray
                The local fisher matrix, of shape (n_genes, n_params, n_params).
            - local_gradient: ndarray
                The local gradient, of shape (n_genes, n_params).
            - PQN_diverged_mask: ndarray
                A boolean mask indicating if the gene has diverged in the prox newton
                algorithm, of shape (n_non_zero_genes,).
            - PQN_mask: ndarray
                A boolean mask indicating if the gene should be used for the
                proximal newton step, of shape (n_non_zero_genes,).
            - global_reg_nll: ndarray
                The global regularized nll, of shape (n_non_zero_genes,).
            - newton_decrement_on_mask: Optional[ndarray]
                The newton decrement, of shape (n_ngenes,).
                This is None at the first round of the prox newton algorithm.
            - round_number_PQN: int
                The current round number of the prox newton algorithm.
            - ascent_direction_on_mask: Optional[ndarray]
                The ascent direction, of shape (n_genes, n_params), where
                n_genes is the current number of genes that are active (True
                in the PQN_mask).
            - irls_diverged_mask: ndarray
                A boolean mask indicating if the gene has diverged in the IRLS
                algorithm.

        Returns
        -------
        dict[str, Any]
            A dictionary containing all the necessary info to run the method.
            If we are not at the last iteration, it contains the following fields:
            - beta: ndarray
                The log fold changes, of shape (n_non_zero_genes, n_params).
            - PQN_mask: ndarray
                A boolean mask indicating if the gene should be used for the
                proximal newton step.
                It is of shape (n_non_zero_genes,)
            - PQN_diverged_mask: ndarray
                A boolean mask indicating if the gene has diverged in the prox newton
                algorithm. It is of shape (n_non_zero_genes,)
            - ascent_direction_on_mask: np.ndarray
                The ascent direction, of shape (n_genes, n_params), where
                n_genes is the current number of genes that are active (True
                in the PQN_mask).
            - newton_decrement_on_mask: np.ndarray
                The newton decrement, of shape (n_ngenes,).
            - round_number_PQN: int
                The current round number of the prox newton algorithm.
            - global_reg_nll: ndarray
                The global regularized nll, of shape (n_non_zero_genes,).
            - irls_diverged_mask: ndarray
                A boolean mask indicating if the gene has diverged in the IRLS
                algorithm.
            If we are at the last iteration, it contains the following fields:
            - beta: ndarray
                The log fold changes, of shape (n_non_zero_genes, n_params).
            - PQN_diverged_mask: ndarray
                A boolean mask indicating if the gene has diverged in the prox newton
                algorithm. It is of shape (n_non_zero_genes,)
            - irls_diverged_mask: ndarray
                A boolean mask indicating if the gene has diverged in the IRLS
                algorithm.
        """
        # Wwe use the following naming convention: when we say "on mask", we mean
        # that we restrict the quantity to the genes that are active in the proximal
        # newton
        # algorithm. We therefore need to ensure that these quantities are readjusted
        # when we change the proximal quasi newton mask.

        # Load main params from the first state
        beta = shared_states[0]["beta"]
        PQN_diverged_mask = shared_states[0]["PQN_diverged_mask"]
        PQN_mask = shared_states[0]["PQN_mask"]
        reg_nll = shared_states[0]["global_reg_nll"]
        ascent_direction_on_mask = shared_states[0]["ascent_direction_on_mask"]
        newton_decrement_on_mask = shared_states[0]["newton_decrement_on_mask"]
        round_number_PQN = shared_states[0]["round_number_PQN"]

        reg_parameter = 1e-6

        # ---- Step 0: Aggregate the nll, gradient and fisher info ---- #

        new_fisher_options_on_mask = sum(
            [state["local_fisher"] for state in shared_states]
        )

        new_gradient_options_on_mask = sum(
            [state["local_gradient"] for state in shared_states]
        )
        new_reg_nll_options_on_mask = sum(
            [state["local_nll"] for state in shared_states]
        )

        # ---- Step 1: Add the regularization term ---- #

        # ---- Step 1a: Compute the new beta options ---- #

        # In order to regularize, we have to compute the beta values at which
        # the nll, gradient and fisher informations were evaluated in the local steps.

        beta_on_mask = beta[PQN_mask]

        if round_number_PQN == 0:
            # In this case, there is no line search, and only
            # beta is considered in the local steps.
            new_beta_options_on_mask = beta_on_mask[None, :]

        else:
            # In this case, there is a line search, and we have to
            # compute the new beta options
            assert ascent_direction_on_mask is not None
            step_sizes = 0.5 ** np.arange(self.PQN_num_iters_ls)
            new_beta_options_on_mask = np.clip(
                beta_on_mask[None, :, :]
                - step_sizes[:, None, None] * ascent_direction_on_mask[None, :, :],
                -self.max_beta,
                self.max_beta,
            )

        # ---- Step 1b: Add the regularization ---- #

        # Add a regularization term to fisher info

        if new_fisher_options_on_mask is not None:
            # Add the regularization term to construct the Fisher info with prior
            # from the Fisher info without prior
            cross_term = (
                new_gradient_options_on_mask[:, :, :, None]
                @ new_beta_options_on_mask[:, :, None, :]
            )
            beta_term = (
                new_beta_options_on_mask[:, :, :, None]
                @ new_beta_options_on_mask[:, :, None, :]
            )
            new_fisher_options_on_mask += (
                reg_parameter * (cross_term + cross_term.transpose(0, 1, 3, 2))
                + reg_parameter**2 * beta_term
            )

            # Furthermore, add a ridge term to the Fisher info for numerical stability
            # This factor decreases log linearly between and initial and final reg
            # The decreasing factor is to ensure that the first steps correspond to
            # gradient descent steps, as we are too far from the optimum
            # to use the Fisher info.
            # Note that other schemes seem to work as well: 1 for 20 iterations then
            # 1e-6
            # 1 for 20 iterations then 1e-2 (to confirm), or 1 for 20 iterations and
            # then
            # 1/n_samples.
            initial_reg_fisher = 1
            final_reg_fisher = 1e-6
            reg_fisher = initial_reg_fisher * (
                final_reg_fisher / initial_reg_fisher
            ) ** (round_number_PQN / self.PQN_num_iters)

            new_fisher_options_on_mask = (
                new_fisher_options_on_mask
                + np.diag(np.repeat(reg_fisher, new_fisher_options_on_mask.shape[-1]))[
                    None, None, :, :
                ]
            )

        # Add regularization term to gradient
        new_gradient_options_on_mask += reg_parameter * new_beta_options_on_mask

        # Add regularization term to the nll
        new_reg_nll_options_on_mask += (
            0.5 * reg_parameter * np.sum(new_beta_options_on_mask**2, axis=2)
        )

        # ---- Step 2: Compute best step size, and new values for this step size ---- #

        # This is only done if we are not at the first round of the prox newton
        # algorithm, as the first rounds serves only to evaluate the nll, gradient
        # and fisher info at the current beta values, and compute the first
        # ascent direction.

        if round_number_PQN > 0:
            # ---- Step 2a: See which step sizes pass the selection criteria ---- #

            assert reg_nll is not None
            reg_nll_on_mask = reg_nll[PQN_mask]

            obj_diff_options_on_mask = (
                reg_nll_on_mask[None, :] - new_reg_nll_options_on_mask
            )  # of shape n_steps, n_PQN_genes

            step_sizes = 0.5 ** np.arange(self.PQN_num_iters_ls)

            # Condition 1: Armijo condition
            # This condition is also called the first Wolfe condition.
            # Reference https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
            admissible_step_size_options_mask = (
                obj_diff_options_on_mask
                >= self.PQN_c1 * step_sizes[:, None] * newton_decrement_on_mask[None, :]
            )

            # ---- Step 2b: Identify genes that have diverged, and remove them ---- #

            # For each gene, we check if there is at least one step size that satisfies
            # the selection criteria. If there is none, we consider that the gene has
            # diverged: we remove all such genes from the PQN_mask
            # and add them to the PQN_diverged_mask.

            diverged_gene_mask_in_current_PQN_mask = np.all(
                ~admissible_step_size_options_mask, axis=0
            )

            # Remove these diverged genes for which we cannot find
            # a correct step size

            PQN_diverged_mask[PQN_mask] = diverged_gene_mask_in_current_PQN_mask
            PQN_mask[PQN_mask] = ~diverged_gene_mask_in_current_PQN_mask

            # Restrict all the quantities defined on the prox newton
            # mask to the new prox newton mask

            obj_diff_options_on_mask = obj_diff_options_on_mask[
                :, ~diverged_gene_mask_in_current_PQN_mask
            ]
            reg_nll_on_mask = reg_nll_on_mask[~diverged_gene_mask_in_current_PQN_mask]
            beta_on_mask = beta_on_mask[~diverged_gene_mask_in_current_PQN_mask]
            admissible_step_size_options_mask = admissible_step_size_options_mask[
                :, ~diverged_gene_mask_in_current_PQN_mask
            ]
            new_reg_nll_options_on_mask = new_reg_nll_options_on_mask[
                :, ~diverged_gene_mask_in_current_PQN_mask
            ]
            new_gradient_options_on_mask = new_gradient_options_on_mask[
                :, ~diverged_gene_mask_in_current_PQN_mask, :
            ]
            new_beta_options_on_mask = new_beta_options_on_mask[
                :, ~diverged_gene_mask_in_current_PQN_mask, :
            ]

            new_fisher_options_on_mask = new_fisher_options_on_mask[
                :, ~diverged_gene_mask_in_current_PQN_mask, :, :
            ]

            # ---- Step 2c: Find the best step size for each gene ---- #

            # Here, we find the best step size for each gene that satisfies the
            # selection criteria (i.e. the largest).
            # We do this by finding the first index for which
            # the admissible step size mask is True.
            # We then create the new beta, gradient, fisher info and reg nll by
            # taking the option corresponding to the best step size

            new_step_size_index = np.argmax(admissible_step_size_options_mask, axis=0)
            arange_PQN = np.arange(len(new_step_size_index))

            new_beta_on_mask = new_beta_options_on_mask[new_step_size_index, arange_PQN]
            new_gradient_on_mask = new_gradient_options_on_mask[
                new_step_size_index, arange_PQN
            ]
            new_fisher_on_mask = new_fisher_options_on_mask[
                new_step_size_index, arange_PQN
            ]

            obj_diff_on_mask = obj_diff_options_on_mask[new_step_size_index, arange_PQN]

            new_reg_nll_on_mask = new_reg_nll_options_on_mask[
                new_step_size_index, arange_PQN
            ]

            # ---- Step 2d: Update the beta values and the reg_nll values ---- #

            beta[PQN_mask] = new_beta_on_mask
            reg_nll[PQN_mask] = new_reg_nll_on_mask

            # ---- Step 2e: Check for convergence of the method ---- #

            convergence_mask = (
                np.abs(obj_diff_on_mask)
                / (
                    np.maximum(
                        np.maximum(
                            np.abs(new_reg_nll_on_mask),
                            np.abs(reg_nll_on_mask),
                        ),
                        1,
                    )
                )
                < self.PQN_ftol
            )

            # ---- Step 2f: Remove converged genes from the mask ---- #
            PQN_mask[PQN_mask] = ~convergence_mask

            # If we reach the max number of iterations, we stop
            if round_number_PQN == self.PQN_num_iters:
                # In this case, we are finished.
                return {
                    "beta": beta,
                    "PQN_diverged_mask": PQN_diverged_mask | PQN_mask,
                    "irls_diverged_mask": shared_states[0]["irls_diverged_mask"],
                }

            # We restrict all quantities to the new mask

            new_gradient_on_mask = new_gradient_on_mask[~convergence_mask]
            new_beta_on_mask = new_beta_on_mask[~convergence_mask]
            new_fisher_on_mask = new_fisher_on_mask[~convergence_mask]

            # Note, this is the old beta
            beta_on_mask = beta_on_mask[~convergence_mask]

        else:
            # In this case, we are at the first round of the prox newton algorithm
            # In this case, we simply instantiate the new values to the first
            # values that were computed in the local steps, to be able to compute
            # the first ascent direction.
            beta_on_mask = None
            new_gradient_on_mask = new_gradient_options_on_mask[0]
            new_beta_on_mask = new_beta_options_on_mask[0]
            new_fisher_on_mask = new_fisher_options_on_mask[0]

            # Set the nll
            reg_nll[PQN_mask] = new_reg_nll_options_on_mask[0]

        # ---- Step 3: Compute the gradient scaling matrix ---- #

        gradient_scaling_matrix_on_mask = compute_gradient_scaling_matrix_fisher(
            fisher=new_fisher_on_mask,
            backend=self.joblib_backend,
            num_jobs=self.num_jobs,
            joblib_verbosity=self.joblib_verbosity,
            batch_size=self.irls_batch_size,
        )

        # ---- Step 4: Compute the ascent direction and the newton decrement ---- #

        (
            ascent_direction_on_mask,
            newton_decrement_on_mask,
        ) = compute_ascent_direction_decrement(
            gradient_scaling_matrix=gradient_scaling_matrix_on_mask,
            gradient=new_gradient_on_mask,
            beta=new_beta_on_mask,
            max_beta=self.max_beta,
        )

        round_number_PQN += 1

        return {
            "beta": beta,
            "PQN_mask": PQN_mask,
            "PQN_diverged_mask": PQN_diverged_mask,
            "ascent_direction_on_mask": ascent_direction_on_mask,
            "newton_decrement_on_mask": newton_decrement_on_mask,
            "round_number_PQN": round_number_PQN,
            "global_reg_nll": reg_nll,
            "irls_diverged_mask": shared_states[0]["irls_diverged_mask"],
        }
