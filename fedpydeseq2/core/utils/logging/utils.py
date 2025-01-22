import json
from pathlib import Path
from typing import Any

from loguru import logger


def setup_workflow_file(log_config_path: str | Path) -> None:
    """Create the workflow file if the configuration is set to True.

    Parameters
    ----------
    log_config_path : str or Path
        The path to the log configuration file.
    """
    log_config_path = Path(log_config_path)
    if not log_config_path.exists():
        return
    with log_config_path.open("r") as log_config_file:
        log_config = json.load(log_config_file)

    workflow_config = log_config.get("generate_workflow")
    if workflow_config is None:
        return
    # Get the create workflow flag
    create_workflow = workflow_config.get("create_workflow")
    if create_workflow is None or not create_workflow:
        return
    # Get the workflow file path
    workflow_file_path = workflow_config.get("workflow_file_path")
    # If it is not provided, return
    if workflow_file_path is None:
        return
    # Create parent directories and the file
    workflow_file_path = Path(workflow_file_path)
    workflow_file_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_file_path.touch(exist_ok=True)

    clean_workflow = workflow_config.get("clean_workflow_files")
    if clean_workflow is None or not clean_workflow:
        return
    # Clean the workflow file
    with workflow_file_path.open("w") as workflow_file:
        workflow_file.write("")
    return


def set_log_config_path(log_config_path: str | Path | None) -> None:
    """Create a log_config_path.json in the same directory as this Python file.

    The JSON file contains one key: 'log_config_path' which is set to null if
    `log_config_path` is None, or the string representation of the path if a
    path is provided.
    If the file already exists, a warning is raised using loguru, but the file
    is overwritten.

    Parameters
    ----------
    log_config_path : str or Path or None
        The path to be set in the JSON configuration file. If None, the value
        will be null.

    Raises
    ------
    None
    """
    # Determine the directory of the current file
    current_dir = Path(__file__).parent

    # Define the path to the log_config_path.json file
    config_file_path = current_dir / "log_config_path.json"

    # Check if the file already exists and log a warning if it does
    if config_file_path.exists():
        logger.warning(f"{config_file_path} already exists and will be overwritten.")

    # Prepare the content to be written to the JSON file
    config_content = {
        "log_config_path": str(log_config_path) if log_config_path is not None else None
    }

    # Write the content to the JSON file
    with config_file_path.open("w") as config_file:
        json.dump(config_content, config_file, indent=4)

    # Set up the workflow file
    if log_config_path is not None:
        setup_workflow_file(log_config_path)


def read_log_config_path() -> dict[str, Any] | None:
    """Read the log_config_path.json file and returns its content as a dictionary.

    Returns
    -------
    dict or None
        The content of the log_config_path.json file as a dictionary,
        or None if the file does not exist or the path is null.
    """
    current_dir = Path(__file__).parent
    config_file_path = current_dir / "log_config_path.json"

    if not config_file_path.exists():
        return None

    with config_file_path.open("r") as config_file:
        config_content = json.load(config_file)

    log_config_path = config_content.get("log_config_path")
    if log_config_path is None:
        return None

    log_config_path = Path(log_config_path)
    if not log_config_path.exists():
        raise FileNotFoundError(
            f"The log configuration file at {log_config_path} does not exist."
        )

    with log_config_path.open("r") as log_config_file:
        log_config = json.load(log_config_file)
        return log_config


def get_logger_configuration() -> str | None:
    """Return the logger configuration ini path from the log configuration file.

    Returns
    -------
    str or None
        The logger configuration ini path, or None if not available.
    """
    config = read_log_config_path()
    if config is None:
        return None
    return config.get("logger_configuration_ini_path")


def get_workflow_configuration() -> dict[str, Any] | None:
    """Return the generate workflow dictionary from the log configuration file.

    Returns
    -------
    dict or None
        The generate workflow dictionary, or None if not available.
    """
    config = read_log_config_path()
    if config is None:
        return None
    return config.get("generate_workflow")


def log_shared_state_adata_flag() -> bool | None:
    """Return the log_adata_content flag from the log configuration file.

    Returns
    -------
    bool or None
        The log_adata_content flag, or None if not available.
    """
    config = read_log_config_path()
    if config is None:
        return None
    return config.get("log_shared_state_adata_content")


def log_shared_state_size_flag() -> bool | None:
    """Return the log_shared_state_size flag from the log configuration file.

    Returns
    -------
    bool or None
        The log_shared_state_size flag, or None if not available.
    """
    config = read_log_config_path()
    if config is None:
        return None
    return config.get("log_shared_state_size")


def get_workflow_file() -> Path | None:
    """Return the workflow file path if the configuration is set to True.

    Returns
    -------
    Path or None
        The workflow file path, or None if not available.
    """
    workflow_config = get_workflow_configuration()
    if workflow_config is not None:
        create_workflow = workflow_config.get("create_workflow")
        if create_workflow:
            workflow_file = workflow_config.get("workflow_file_path")
            assert workflow_file is not None, "Workflow file path is not provided."
            workflow_file_path = Path(workflow_file)
            assert workflow_file_path.exists(), "Workflow file does not exist."
            return workflow_file_path
    return None
