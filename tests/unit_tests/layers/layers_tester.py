"""Module implenting a tester class for the reconstruct_adatas decorator."""

import pickle as pkl
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from substrafl import ComputePlanBuilder
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.remote import remote_data

from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from tests.unit_tests.layers.utils import create_dummy_adata_with_layers


class SimpleLayersTester(
    ComputePlanBuilder,
):
    """A tester class for the reconstruct_adatas decorator.

    This class implements the following steps.
    First, it creates a dummy AnnData object with the necessary layers, varm, obsm
    obs to be able to reconstruct all simple layers.

    We then set this dummy AnnData object as the local_adata and local_reference_adata
    attributes of the class.

    We then test the reconstruct_adatas decorator which for now only works on the
    local_adata # TODO add the refit adata once it is done
    by performing the following steps:

    1. We perform an empty local step with the reconstruct_adatas decorator.
    This ensures
    that we have applied the decorator.

    2. We perform a check without the decorator. In this check, we check the state
    of the local_adata before the decorator is applied.
    In the case where the
    save_layers_to_disk attribute is set to False, we check that
    the the local_adata contains no layers nor counts, except for the layers that
    have been set in the layers_to_save attribute. If the cooks layer was present in
    the reference adata, we check that it is still present in the local_adata, as the
    decorator should not touch it.
    We check that the values of the layers are the same.

    In the case where the save_layers_to_disk attribute is set to True, or that this
    attribute does not exist at all we check that all layers that were present in the
    reference adata are still present in the local_adata, and that the layers that were
    not present are still not present. We also check that the values of the layers are
    the same.

    3. We perform a check with the decorator. In this check, we check the state of the
    local_adata after the decorator is applied once more.
    In the case where the save_layers_to_disk attribute is set to False, the
    local adata must contain all the layers that are present in the layers_to_save
    attribute if it exists, the cooks layer if it was present in the reference adata,
    and all the layers that are present in the layers_to_load attribute if it exists.
    If the layers_to_load attribute does not exist, we check that all layers that were
    present in the reference adata are still present in the local_adata.
    We also check that
    the values of the layers are the same. We check that the counts are present and
    that they are equal.

    In the case where the save_layers_to_disk attribute is set to True, or that this
    attribute does not exist at all we check that all layers that were present in the
    reference adata are still present in the local_adata, and that the layers that were
    not present are still not present. We also check that the values of the layers are
    the same. We check that the counts are present and that they are equal.

    Parameters
    ----------
    num_row_values_equal_n_params : bool, optional
        Whether the number of values taken by the design is equal to
        the number of parameters.
        If False, the number of row values is equal to the number of parameters + 1.
        Defaults to False.

    add_cooks : bool, optional
        Whether to add the cooks layer. Defaults to False.

    n_params : int, optional
        Number of parameters. Defaults to 5.

    has_save_layers_to_disk_attribute : bool, optional
        Whether the class has the save_layers_to_disk attribute. Defaults to False.

    save_layers_to_disk : bool, optional
        Whether to save the layers to disk. Defaults to False.

    has_layers_to_save_on_disk_attribute : bool, optional
        Whether the class has the layers_to_save_on_disk attribute.

    test_layers_to_save_on_disk_attribute: bool
        Whether the layers_to_save_on_disk contains layers or not.

    test_layers_to_save : bool, optional
        Whether to test the layers to save. Defaults to False.

    test_layers_to_load : bool, optional
        Whether to test the layers to load. Defaults to False.
    """

    def __init__(
        self,
        num_row_values_equal_n_params=False,
        add_cooks=False,
        save_cooks=False,
        n_params=5,
        has_save_layers_to_disk_attribute: bool = False,
        save_layers_to_disk: bool = False,
        has_layers_to_save_on_disk_attribute: bool = False,
        test_layers_to_save_on_disk_attribute: bool = True,
        test_layers_to_save: bool = False,
        test_layers_to_load: bool = False,
        *args,
        **kwargs,
    ):
        # Add all arguments to super init so that they can be retrieved by nodes.
        super().__init__(
            num_row_values_equal_n_params=num_row_values_equal_n_params,
            add_cooks=add_cooks,
            n_params=n_params,
            has_save_layers_to_disk_attribute=has_save_layers_to_disk_attribute,
            save_layers_to_disk=save_layers_to_disk,
            has_layers_to_save_on_disk_attribute=has_layers_to_save_on_disk_attribute,
            test_layers_to_save=test_layers_to_save,
            test_layers_to_load=test_layers_to_load,
        )

        #### Quantities needed to create the adata
        if not add_cooks:
            assert not save_cooks, "Cannot save Cooks layer if Cooks not added."

        self.num_row_values_equal_n_params = num_row_values_equal_n_params
        self.add_cooks = add_cooks
        self.n_params = n_params

        #### Quantities needed to save and load the layers

        if has_save_layers_to_disk_attribute:
            self.save_layers_to_disk = save_layers_to_disk

        self.has_layers_to_save_on_disk_attributes = (
            has_layers_to_save_on_disk_attribute
        )
        self.test_layers_to_save = test_layers_to_save
        self.test_layers_to_load = test_layers_to_load
        self.save_cooks = save_cooks
        if (
            has_layers_to_save_on_disk_attribute
            and test_layers_to_save_on_disk_attribute
        ):
            self.layers_to_save_on_disk = {
                "local_adata": ["_fit_lin_mu_hat", "cooks"]
                if save_cooks
                else ["_fit_lin_mu_hat"],
                "refit_adata": None,
            }
        elif save_cooks:
            self.layers_to_save_on_disk = {
                "local_adata": ["cooks"],
                "refit_adata": None,
            }

        self.local_adata: ad.AnnData | None = None
        self.refit_adata: ad.AnnData | None = None
        self.local_reference_adata: ad.AnnData | None = None

    def build_compute_plan(
        self,
        train_data_nodes: list[TrainDataNode],
        aggregation_node: AggregationNode,
        evaluation_strategy=None,
        num_rounds=None,
        clean_models=True,
    ):
        """Build the computation graph to run the test of the decorator.

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

        local_states, shared_states, round_idx = local_step(
            local_method=self.init_local_states,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=None,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Initialize local states",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        if self.test_layers_to_save:
            layers_to_save_on_disk = {
                "local_adata": ["_irls_mu_hat"],
                "refit_adata": None,
            }
        else:
            layers_to_save_on_disk = None
        if self.test_layers_to_load:
            layers_to_load = {
                "local_adata": ["_mu_hat"]
                if not self.save_cooks
                else ["_mu_hat", "cooks"],
                "refit_adata": None,
            }
        else:
            layers_to_load = None

        local_states, _, round_idx = local_step(
            local_method=self.empty_remote_data_method,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=None,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Empty remote data method",
            round_idx=round_idx,
            clean_models=clean_models,
            method_params={
                "layers_to_save_on_disk": layers_to_save_on_disk,
                "layers_to_load": layers_to_load,
            },
        )

        local_states, _, round_idx = local_step(
            local_method=self.perform_check_without_decorator,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=None,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Check without decorator",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        local_step(
            local_method=self.perform_check_with_decorator,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=None,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Check with decorator",
            round_idx=round_idx,
            clean_models=clean_models,
            method_params={
                "layers_to_save_on_disk": layers_to_save_on_disk,
                "layers_to_load": layers_to_load,
            },
        )

    @remote_data
    def init_local_states(
        self, data_from_opener: ad.AnnData, shared_state: Any
    ) -> None:
        """Initialize the local states of the strategy.

        This method creates a dummy AnnData object with the necessary layers, varm, obsm
        obs to be able to reconstruct all simple layers. We load the
        counts from the data_from_opener.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener.

        shared_state : Any
            Shared state. Not used.
        """
        dummy_adata = create_dummy_adata_with_layers(
            data_from_opener=data_from_opener,
            num_row_values_equal_n_params=self.num_row_values_equal_n_params,
            add_cooks=self.add_cooks,
            n_params=self.n_params,
        )

        self.local_adata = dummy_adata.copy()
        self.refit_adata = None
        self.local_reference_adata = dummy_adata.copy()

    @remote_data
    @reconstruct_adatas
    def empty_remote_data_method(
        self, data_from_opener: ad.AnnData, shared_state: Any
    ) -> dict:
        """Empty method to test the reconstruct_adatas decorator.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : Any
            Shared state. Not used.

        Returns
        -------
        dict
            An empty dictionary.
        """
        return {}

    @remote_data
    def perform_check_without_decorator(
        self, data_from_opener: ad.AnnData, shared_state: Any
    ):
        """Check the state of the local adata before the decorator is applied.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : Any
            Shared state. Not used.
        """

        all_layers_saved = check_if_all_layers_saved(self)
        if all_layers_saved:
            return

        # From now on, the save_layers_to_disk attribute exists and is False

        # The layers that should be present are the layers that are either in
        # the layers_to_save_on_disk attribute, or the ones that were fed
        # to the method params.

        should_be_present = {"local_adata": [], "refit_adata": []}

        # We look at the layers that were defined to be saved globally.
        if (
            hasattr(self, "layers_to_save_on_disk")
            and self.layers_to_save_on_disk is not None
        ):
            for adata_name in {"local_adata", "refit_adata"}:
                to_save_on_disk_global_adata = self.layers_to_save_on_disk[adata_name]
                if to_save_on_disk_global_adata is not None:
                    should_be_present[adata_name].extend(to_save_on_disk_global_adata)

        # If we must save cooks, add them here.
        elif self.save_cooks:
            should_be_present["local_adata"].append("cooks")

        # Add all the layers in the layers_to_save_on_disk method parameter if it exists
        if self.test_layers_to_save:
            should_be_present["local_adata"].append("_irls_mu_hat")

        # Test that all layers that should be present are present, and only those.
        # Check equality on the present layers.

        for adata_name in {"local_adata", "refit_adata"}:
            adata = getattr(self, adata_name)
            should_be_present_adata = should_be_present[adata_name]
            if adata is None:
                assert len(should_be_present_adata) == 0
            else:
                assert set(adata.layers.keys()) == set(should_be_present_adata)
                if adata_name == "refit_adata":
                    continue  # TODO implement check once we really implement the test
                for layer_name in list(should_be_present_adata):
                    assert np.allclose(
                        self.local_reference_adata.layers[layer_name],
                        adata.layers[layer_name],
                        equal_nan=True,
                    )
        # TODO do the corresponding check for the refit adata.
        assert self.local_adata.X is None

    @remote_data
    @reconstruct_adatas
    def perform_check_with_decorator(
        self, data_from_opener: ad.AnnData, shared_state: Any
    ):
        """Check the state of the local adata after the decorator is applied.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used. (but used by the decorator)
        shared_state : Any
            Shared state. Not used.
        """

        all_layers_saved = check_if_all_layers_saved(self)
        if all_layers_saved:
            return

        # Now we assumed that all layers are not saved to disk.
        should_be_loaded = {"local_adata": [], "refit_adata": []}
        if self.test_layers_to_load:
            # In that case, we only have one layer to load
            should_be_loaded["local_adata"].append("_mu_hat")
            if self.save_cooks:
                should_be_loaded["local_adata"].append("cooks")

        else:
            # If we save cooks, then cooks should be loaded. if not, then not.
            should_be_loaded["local_adata"] = list(
                self.local_reference_adata.layers.keys()
            )
            if not self.save_cooks:
                if "cooks" in should_be_loaded["local_adata"]:
                    should_be_loaded["local_adata"].remove("cooks")

        for adata_name in {"local_adata", "refit_adata"}:
            adata = getattr(self, adata_name)
            should_be_loaded_adata = should_be_loaded[adata_name]
            if adata is None:
                assert len(should_be_loaded_adata) == 0
            else:
                # It is only a subset because some sub layers can be created in the
                # meantime.

                assert set(should_be_loaded_adata).issubset(set(adata.layers.keys()))
                assert adata_name == "local_adata"  # TODO check for refit
                # Check the equality
                for layer_name in adata.layers.keys():
                    assert np.allclose(
                        self.local_reference_adata.layers[layer_name],
                        adata.layers[layer_name],
                        equal_nan=True,
                    )

        assert np.allclose(
            self.local_reference_adata.X,
            self.local_adata.X,
        )

    def save_local_state(self, path: Path) -> None:
        """Save the local state of the strategy.

        Parameters
        ----------
        path : Path
            Path to the file where to save the state. Automatically handled by subtrafl.
        """
        state_to_save = {
            "local_adata": self.local_adata,
            "refit_adata": self.refit_adata,
            "local_reference_adata": self.local_reference_adata,
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

        self.local_adata = state_to_load["local_adata"]
        self.refit_adata = state_to_load["refit_adata"]
        self.local_reference_adata = state_to_load["local_reference_adata"]
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


def check_if_all_layers_saved(self):
    """

    Parameters
    ----------
    self

    Returns
    -------

    """
    if not hasattr(self, "save_layers_to_disk") or self.save_layers_to_disk:
        # Check that all layers originally present are still present,
        assert set(self.local_reference_adata.layers.keys()) == set(
            self.local_adata.layers.keys()
        )
        # and that the layers that were not present are still not present
        for layer_name in self.local_reference_adata.layers.keys():
            assert np.allclose(
                self.local_reference_adata.layers[layer_name],
                self.local_adata.layers[layer_name],
                equal_nan=True,
            )
        return True
    return False
