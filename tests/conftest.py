import json
from pathlib import Path

import fedpydeseq2_datasets
import pytest

DEFAULT_LOGGING_CONFIGURATION_PATH: Path | None = None


def get_default_logging_path_from_json() -> Path | None:
    """Get the default logging configuration path from the paths.json file.

    Returns
    -------
    Path or None
        The path to the default logging configuration file if found.
    """
    specified_paths = Path(__file__).parent / "paths.json"
    if specified_paths.exists():
        with open(specified_paths) as f:
            json_data = json.load(f)
            if "default_logging_config" in json_data:
                logging_config_path = json_data["default_logging_config"]
                if logging_config_path is None:
                    return None
                if logging_config_path.startswith("/"):
                    return Path(logging_config_path)
                return Path(__file__).parent / logging_config_path
    return None


def get_workflow_logging_path_from_json() -> Path | None:
    """Get the workflow logging configuration path from the paths.json file.

    Returns
    -------
    Path or None
        The path to the workflow logging configuration file if found.
    """
    specified_paths = Path(__file__).parent / "paths.json"
    if specified_paths.exists():
        with open(specified_paths) as f:
            json_data = json.load(f)
            if "workflow_logging_config" in json_data:
                logging_config_path = json_data["workflow_logging_config"]
                if logging_config_path is None:
                    return None
                if logging_config_path.startswith("/"):
                    return Path(logging_config_path)
                return Path(__file__).parent / logging_config_path
    return None


def get_logger_default_configuration_file() -> str:
    """Get the default logger configuration file.

    Returns
    --------
    str
        The path to the default logger configuration file.
    """
    current_dir = Path(__file__).parent
    return str(current_dir / "default_test_logger_configuration.ini")


@pytest.fixture(scope="session", autouse=True)
def set_default_logging_config_path(tmpdir_factory):
    """Set the DEFAULT_LOGGING_CONFIGURATION_PATH to the default configuration file."""
    global DEFAULT_LOGGING_CONFIGURATION_PATH
    # If the default logging configuration is specified in the paths.json file
    # we use that path.
    logging_path_from_json = get_default_logging_path_from_json()
    if logging_path_from_json is not None:
        DEFAULT_LOGGING_CONFIGURATION_PATH = logging_path_from_json
        yield
        return

    # Otherwise, we create a temporary logging configuration file,
    # with the default logger configuration file.
    # This configuration file will logs the content
    # of the shared state adata, but not the size.
    # It does not generate the workflow either.

    logger_config = get_logger_default_configuration_file()

    logging_config = {
        "logger_configuration_ini_path": logger_config,
        "generate_workflow": {
            "create_workflow": False,
            "workflow_file_path": None,
            "clean_workflow_file": False,
        },
        "log_shared_state_adata_content": True,
        "log_shared_state_size": False,
    }

    # Create the directory and the logging configuration file
    tmp_logging_directory = Path(tmpdir_factory.mktemp("logging"))
    tmp_logging_config_path = tmp_logging_directory / "default_logging_config.json"
    with tmp_logging_config_path.open("w") as f:
        json.dump(logging_config, f)

    # Set the global variable
    DEFAULT_LOGGING_CONFIGURATION_PATH = tmp_logging_config_path
    yield
    return


@pytest.fixture(scope="session")
def workflow_logging_configuration_path(tmpdir_factory) -> Path:
    """Fixture to get the path to the workflow logging configuration file."""
    # Check if the workflow logging configuration is specified in the paths.json file
    logging_path_from_json = get_workflow_logging_path_from_json()
    if logging_path_from_json is not None:
        return logging_path_from_json

    # If not specified, we create a temporary logging configuration file,
    # with the default logger configuration file, as well as
    # a temporary workflow file.

    logging_directory = Path(tmpdir_factory.mktemp("logging"))
    workflow_file_path = logging_directory / "workflow.txt"
    logger_config = get_logger_default_configuration_file()

    logging_config = {
        "logger_configuration_ini_path": logger_config,
        "generate_workflow": {
            "create_workflow": True,
            "workflow_file_path": str(workflow_file_path),
            "clean_workflow_file": True,
        },
        "log_shared_state_adata_content": False,
        "log_shared_state_size": False,
    }

    logging_config_path = logging_directory / "workflow_logging_configuration.json"
    with logging_config_path.open("w") as f:
        json.dump(logging_config, f)
    return logging_config_path


@pytest.fixture(scope="session")
def raw_data_path():
    """Fixture to get the path to the raw data."""
    default_paths = Path(__file__).parent / "paths_default.json"
    specified_paths = Path(__file__).parent / "paths.json"
    if specified_paths.exists():
        with open(specified_paths) as f:
            raw_data_path = json.load(f)["raw_data"]
    else:
        with open(default_paths) as f:
            raw_data_path = json.load(f)["raw_data"]
    if raw_data_path.startswith("/"):
        raw_data_path = Path(raw_data_path)
    else:
        raw_data_path = Path(__file__).parent / raw_data_path
    print("Test raw data path")
    return raw_data_path


@pytest.fixture(scope="session")
def tmp_processed_data_path(tmpdir_factory):
    return Path(tmpdir_factory.mktemp("processed"))


@pytest.fixture(scope="session")
def tcga_assets_directory():
    specified_paths = Path(__file__).parent / "paths.json"
    if specified_paths.exists():
        with open(specified_paths) as f:
            json_data = json.load(f)
            if "assets_tcga" in json_data:
                tcga_assets_directory = json_data["assets_tcga"]
                if tcga_assets_directory.startswith("/"):
                    return Path(tcga_assets_directory)
                return Path(__file__).parent / tcga_assets_directory

    fedpydeseq2_datasets_dir = Path(fedpydeseq2_datasets.__file__).parent
    return fedpydeseq2_datasets_dir / "assets/tcga"


@pytest.fixture(scope="session")
def local_processed_data_path():
    default_paths = Path(__file__).parent / "paths_default.json"
    specified_paths = Path(__file__).parent / "paths.json"
    found = False
    if specified_paths.exists():
        with open(specified_paths) as f:
            json_data = json.load(f)
            if "processed_data" in json_data:
                found = True
                processed_data_path = json_data["processed_data"]
    if not found:
        with open(default_paths) as f:
            processed_data_path = json_data["processed_data"]
    if processed_data_path.startswith("/"):
        processed_data_path = Path(processed_data_path)
    else:
        processed_data_path = Path(__file__).parent / processed_data_path
    return processed_data_path
