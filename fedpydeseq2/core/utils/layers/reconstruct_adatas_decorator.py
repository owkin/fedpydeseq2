"""Module containing a decorator to handle simple layers.

This wrapper is used to load and save simple layers from the adata object.
These simple layers are defined in SIMPLE_LAYERS.


"""

from collections.abc import Callable
from functools import wraps
from typing import Any

import anndata as ad
import numpy as np

from fedpydeseq2.core.utils.layers.build_refit_adata import set_basic_refit_adata
from fedpydeseq2.core.utils.layers.build_refit_adata import (
    set_imputed_counts_refit_adata,
)
from fedpydeseq2.core.utils.layers.joblib_utils import get_joblib_parameters
from fedpydeseq2.core.utils.layers.utils import get_available_layers
from fedpydeseq2.core.utils.layers.utils import load_layers
from fedpydeseq2.core.utils.layers.utils import remove_layers

LayersToLoadSaveType = dict[str, list[str] | None] | None


def reconstruct_adatas(method: Callable):
    """Decorate a method to load layers and remove them before saving the state.

    This decorator loads the layers from the data_from_opener and the adata
    object before calling the method. It then removes the layers from the adata
    object after the method is called.

    The object self CAN have the following attributes:

    - save_layers_to_disk: if this argument exists or is True, we save all the layers
    on disk, without removing them at the end of each local step. If it is False,
    we remove all layers that must be removed at the end of each local step.
    This argument is prevalent above all others described below.

    - layers_to_save_on_disk: if this argument exists, contains the layers that
    must be saved on disk at EVERY local step. It can be either None (in which
    case the default behaviour is to save no layers) or a dictionary with a refit_adata
    and local_adata key. The associated values contain either None (no layers) or
    a list of layers to save at each step.

    This decorator adds two parameters to each method decorated with it:
    - layers_to_load
    - layers_to_save_on_disk

    If the layers_to_load is None, the default is to load all available layers.
    Else, we only load the layers specified in the layers_to_load argument.

    The layers_to_save_on_disk argument is ADDED to the layers_to_save_on_disk attribute
    of self for the duration of the method and then removed. That way, the inner
    method can access the names of the layers_to_save_on_disk which will effectively
    be saved at the end of the step.

    Parameters
    ----------
    method : Callable
        The method to decorate. This method is expected to have the following signature:
        method(self, data_from_opener: ad.AnnData, shared_state: Any,
         **method_parameters).

    Returns
    -------
    Callable
        The decorated method, which loads the simple layers before calling the method
        and removes the simple layers after the method is called.

    """

    @wraps(method)
    def method_inner(
        self,
        data_from_opener: ad.AnnData,
        shared_state: Any = None,
        layers_to_load: LayersToLoadSaveType = None,
        layers_to_save_on_disk: LayersToLoadSaveType = None,
        **method_parameters,
    ):
        if layers_to_load is None:
            layers_to_load = {"local_adata": None, "refit_adata": None}
        if hasattr(self, "layers_to_save_on_disk"):
            if self.layers_to_save_on_disk is None:
                global_layers_to_save_on_disk = None
            else:
                global_layers_to_save_on_disk = self.layers_to_save_on_disk.copy()

            if global_layers_to_save_on_disk is None:
                self.layers_to_save_on_disk = {"local_adata": [], "refit_adata": []}
        else:
            self.layers_to_save_on_disk = {"local_adata": [], "refit_adata": []}

        if layers_to_save_on_disk is None:
            layers_to_save_on_disk = {"local_adata": [], "refit_adata": []}

        # Set the layers_to_save_on_disk attribute to the union of the layers specified
        # in the argument and those in the attribute, to be accessed by the method.
        assert isinstance(self.layers_to_save_on_disk, dict)
        for adata_name in ["local_adata", "refit_adata"]:
            if self.layers_to_save_on_disk[adata_name] is None:
                self.layers_to_save_on_disk[adata_name] = []
            if layers_to_save_on_disk[adata_name] is None:
                layers_to_save_on_disk[adata_name] = []
            self.layers_to_save_on_disk[adata_name] = list(
                set(
                    layers_to_save_on_disk[adata_name]
                    + self.layers_to_save_on_disk[adata_name]
                )
            )

        # Check that the layers_to_load and layers_to_save are valid
        assert set(layers_to_load.keys()) == {"local_adata", "refit_adata"}
        assert set(self.layers_to_save_on_disk.keys()) == {"local_adata", "refit_adata"}

        # Load the counts of the adata
        if self.local_adata is not None:
            if self.local_adata.X is None:
                self.local_adata.X = data_from_opener.X

        # Load the available layers
        only_from_disk = (
            not hasattr(self, "save_layers_to_disk") or self.save_layers_to_disk
        )

        # Start by loading the local adata
        check_and_load_layers(
            self, "local_adata", layers_to_load, shared_state, only_from_disk
        )

        # Create the refit adata
        reconstruct_refit_adata_without_layers(self)

        # Load the layers of the refit adata
        check_and_load_layers(
            self, "refit_adata", layers_to_load, shared_state, only_from_disk
        )

        # Apply the method
        shared_state = method(self, data_from_opener, shared_state, **method_parameters)

        # Remove all layers which must not be saved on disk
        for adata_name in ["local_adata", "refit_adata"]:
            adata = getattr(self, adata_name)
            if adata is None:
                continue
            if only_from_disk:
                layers_to_save_on_disk_adata: list | None = list(adata.layers.keys())
            else:
                layers_to_save_on_disk_adata = self.layers_to_save_on_disk[adata_name]
                assert layers_to_save_on_disk_adata is not None
                for layer in layers_to_save_on_disk_adata:
                    if layer not in adata.layers.keys():
                        print("Warning: layer not in adata: ", layer)
            assert layers_to_save_on_disk_adata is not None
            remove_layers(
                adata=adata,
                layers_to_save_on_disk=layers_to_save_on_disk_adata,
                refit=adata_name == "refit_adata",
            )

        # Reset the layers_to_save_on_disk attribute
        try:
            self.layers_to_save_on_disk = global_layers_to_save_on_disk
        except NameError:
            del self.layers_to_save_on_disk

        return shared_state

    return method_inner


def reconstruct_refit_adata_without_layers(self: Any):
    """Reconstruct the refit adata without the layers.

    This function reconstructs the refit adata without the layers.
    It is used to avoid the counts and the obsm being loaded uselessly in the
    refit_adata.

    Parameters
    ----------
    self : Any
        The object containing the adata.

    """
    if self.refit_adata is None:
        return
    if self.local_adata is not None and "replaced" in self.local_adata.varm.keys():
        set_basic_refit_adata(self)
    if self.local_adata is not None and "refitted" in self.local_adata.varm.keys():
        set_imputed_counts_refit_adata(self)


def check_and_load_layers(
    self: Any,
    adata_name: str,
    layers_to_load: dict[str, list[str] | None],
    shared_state: dict | None,
    only_from_disk: bool,
):
    """Check and load layers for a given adata_name.

    This function checks the availability of the layers to load
    and loads them, for the adata_name adata.

    Parameters
    ----------
    self : Any
        The object containing the adata.
    adata_name : str
        The name of the adata to load the layers into.
    layers_to_load : dict[str, Optional[list[str]]]
        The layers to load for each adata. It must have adata_name
        as a key.
    shared_state : Optional[dict]
        The shared state.
    only_from_disk : bool
        Whether to load only the layers from disk.

    """
    adata = getattr(self, adata_name)
    layers_to_load_adata = layers_to_load[adata_name]
    available_layers_adata = get_available_layers(
        adata,
        shared_state,
        refit=adata_name == "refit_adata",
        all_layers_from_disk=only_from_disk,
    )
    if layers_to_load_adata is None:
        layers_to_load_adata = available_layers_adata
    else:
        assert np.all(
            [layer in available_layers_adata for layer in layers_to_load_adata]
        )
    if adata is None:
        return
    assert layers_to_load_adata is not None
    n_jobs, joblib_verbosity, joblib_backend, batch_size = get_joblib_parameters(self)
    load_layers(
        adata=adata,
        shared_state=shared_state,
        layers_to_load=layers_to_load_adata,
        n_jobs=n_jobs,
        joblib_verbosity=joblib_verbosity,
        joblib_backend=joblib_backend,
        batch_size=batch_size,
    )
