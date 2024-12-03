"""
Module containing decorators to log the input and outputs of a method.

All logging is controlled through a logging configuration file.
This configuration file can be either set by the log_config_path attribute of the class,
or by the default_config.ini file in the same directory as this module.
"""
import logging
import logging.config
import os
import pathlib
from collections.abc import Callable
from functools import wraps
from typing import Any

import anndata as ad


def log_save_local_state(method: Callable):
    """
    Decorate a method to log the size of the local state saved.

    This function is destined to decorate the save_local_state method of a class.

    It logs the size of the local state saved in the local state path, in MB.
    This is logged as an info message.

    Parameters
    ----------
    method : Callable
        The method to decorate. This method is expected to have the following signature:
        method(self, path: pathlib.Path).

    Returns
    -------
    Callable
        The decorated method, which logs the size of the local state saved.

    """

    @wraps(method)
    def remote_method_inner(
        self,
        path: pathlib.Path,
    ):
        logger = get_method_logger(self, method)

        output = method(self, path)

        logger.info(
            f"Size of local state saved : "
            f"{os.path.getsize(path) / 1024 / 1024}"
            " MB"
        )

        return output

    return remote_method_inner


def log_remote_data(method: Callable):
    """
    Decorate a remote_data to log the input and outputs.

    This decorator logs the shared state keys with the info level,
    and the different layers of the local_adata and refit_adata with the debug level.

    This is done before and after the method call.

    Parameters
    ----------
    method : Callable
        The method to decorate. This method is expected to have the following signature:
        method(self, data_from_opener: ad.AnnData,
        shared_state: Any = None, **method_parameters).

    Returns
    -------
    Callable
        The decorated method, which logs the shared state keys with the info level
        and the different layers of the local_adata and refit_adata with the debug
        level.
    """

    @wraps(method)
    def remote_method_inner(
        self,
        data_from_opener: ad.AnnData,
        shared_state: Any = None,
        **method_parameters,
    ):
        logger = get_method_logger(self, method)
        logger.info("---- Before running the method ----")
        log_shared_state_adatas(self, method, shared_state)

        shared_state = method(self, data_from_opener, shared_state, **method_parameters)

        logger.info("---- After method ----")
        log_shared_state_adatas(self, method, shared_state)
        return shared_state

    return remote_method_inner


def log_remote(method: Callable):
    """
    Decorate a remote method to log the input and outputs.

    This decorator logs the shared state keys with the info level.

    Parameters
    ----------
    method : Callable
        The method to decorate. This method is expected to have the following signature:
        method(self, shared_states: Optional[list], **method_parameters).

    Returns
    -------
    Callable
        The decorated method, which logs the shared state keys with the info level.

    """

    @wraps(method)
    def remote_method_inner(
        self,
        shared_states: list | None,
        **method_parameters,
    ):
        logger = get_method_logger(self, method)
        if shared_states is not None:
            shared_state = shared_states[0]
            if shared_state is not None:
                logger.info(
                    f"First input shared state keys : {list(shared_state.keys())}"
                )
            else:
                logger.info("First input shared state is None.")
        else:
            logger.info("No input shared states.")

        shared_state = method(self, shared_states, **method_parameters)

        if shared_state is not None:
            logger.info(f"Output shared state keys : {list(shared_state.keys())}")
        else:
            logger.info("No output shared state.")

        return shared_state

    return remote_method_inner


def log_shared_state_adatas(self: Any, method: Callable, shared_state: dict | None):
    """
    Log the information of the local step.

    Precisely, log the shared state keys (info),
    and the different layers of the local_adata and refit_adata (debug).

    Parameters
    ----------
    self : Any
        The class instance
    method : Callable
        The class method.
    shared_state : Optional[dict]
        The shared state dictionary, whose keys we log with the info level.

    """
    logger = get_method_logger(self, method)

    if shared_state is not None:
        logger.info(f"Shared state keys : {list(shared_state.keys())}")
    else:
        logger.info("No shared state")

    for adata_name in ["local_adata", "refit_adata"]:
        if hasattr(self, adata_name) and getattr(self, adata_name) is not None:
            adata = getattr(self, adata_name)
            logger.debug(f"{adata_name} layers : {list(adata.layers.keys())}")
            if "_available_layers" in self.local_adata.uns:
                available_layers = self.local_adata.uns["_available_layers"]
                logger.debug(f"{adata_name} available layers : {available_layers}")
            logger.debug(f"{adata_name} uns keys : {list(adata.uns.keys())}")
            logger.debug(f"{adata_name} varm keys : {list(adata.varm.keys())}")
            logger.debug(f"{adata_name} obsm keys : {list(adata.obsm.keys())}")


def get_method_logger(self: Any, method: Callable) -> logging.Logger:
    """
    Get the method logger from a configuration file.

    If the class instance has a log_config_path attribute,
    the logger is configured with the file at this path.

    Parameters
    ----------
    self: Any
        The class instance
    method: Callable
        The class method.

    Returns
    -------
    logging.Logger
        The logger instance.
    """
    if hasattr(self, "log_config_path"):
        log_config_path = pathlib.Path(self.log_config_path)
    else:
        log_config_path = pathlib.Path(__file__).parent / "default_config.ini"
    logging.config.fileConfig(log_config_path, disable_existing_loggers=False)
    logger = logging.getLogger(method.__name__)
    return logger
