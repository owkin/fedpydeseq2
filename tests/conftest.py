import json
from pathlib import Path

import fedpydeseq2_datasets
import pytest


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
            if "assets_tcga" in json.load(f):
                tcga_assets_directory = json.load(f)["assets_tcga"]
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
            if "processed_data" in json.load(f):
                found = True
                processed_data_path = json.load(f)["processed_data"]
    if not found:
        with open(default_paths) as f:
            processed_data_path = json.load(f)["processed_data"]
    if processed_data_path.startswith("/"):
        processed_data_path = Path(processed_data_path)
    else:
        processed_data_path = Path(__file__).parent / processed_data_path
    return processed_data_path
