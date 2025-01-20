import json
from pathlib import Path


def get_force_clean():
    """Fixture to get the path to the raw data."""
    default_paths = Path(__file__).parent / "logging_paths_template.json"
    specified_paths = Path(__file__).parent / "logging_paths.json"
    if specified_paths.exists():
        with open(specified_paths) as f:
            force_clean = json.load(f)["force_clean"]
    else:
        with open(default_paths) as f:
            force_clean = json.load(f)["force_clean"]
    return force_clean


def get_logging_save_file():
    """Fixture to get the path to the raw data."""
    force_clean = get_force_clean()
    default_paths = Path(__file__).parent / "logging_paths_template.json"
    specified_paths = Path(__file__).parent / "logging_paths.json"
    if specified_paths.exists():
        with open(specified_paths) as f:
            logging_save_file = json.load(f)["logging_save_file"]
    else:
        with open(default_paths) as f:
            logging_save_file = json.load(f)["logging_save_file"]
    if logging_save_file.startswith("/"):
        raw_data_path = Path(logging_save_file)
    else:
        # root of the repo / relative path
        raw_data_path = (
            Path(__file__).parent.parent.parent.parent.parent / logging_save_file
        )
    if force_clean:
        if raw_data_path.exists():
            raw_data_path.unlink()
    # create the directory if it does not exist
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    # create the file if it does not exist
    if not raw_data_path.exists():
        raw_data_path.touch()

    return raw_data_path


LOGGING_SAVE_FILE = get_logging_save_file()
