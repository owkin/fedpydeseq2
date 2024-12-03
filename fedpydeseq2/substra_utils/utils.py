from pathlib import Path

import yaml  # type: ignore
from loguru import logger
from substra import BackendType
from substra import Client
from substrafl.dependency import Dependency


def get_client(
    backend_type: BackendType,
    org_name: str | None = None,
    credentials_path: str | Path | None = None,
) -> Client:
    """
    Return a substra client for a given organization.

    Parameters
    ----------
    backend_type : str
        Name of the backend to connect to. Should be "subprocess", "docker" or "remote"
    org_name : str, optional.
        Name of the organization to connect to. Required when using remote backend.
    credentials_path : str or Path
        Path to the credentials file. By default, will be set to
        Path(__file__).parent / "credentials/credentials.yaml"

    """
    if backend_type not in ("subprocess", "docker", "remote"):
        raise ValueError(
            f"Backend type {backend_type} not supported. Should be one of 'subprocess',"
            f" 'docker' or 'remote'."
        )
    if backend_type == "remote":
        assert (
            org_name is not None
        ), "Organization name must be provided when using remote backend."
        if credentials_path is not None:
            credential_path = Path(credentials_path)
        else:
            credential_path = Path(__file__).parent / "credentials/credentials.yaml"

        with open(credential_path) as file:
            conf = yaml.load(file, Loader=yaml.FullLoader)
        if org_name not in conf.keys():
            raise ValueError(f"Organization {org_name} not found in credentials file.")
        url = conf[org_name]["url"]
        token = conf[org_name]["token"]

        logger.info(
            f"Connecting to {org_name} "
            f"at {url} using credentials "
            f"from {credential_path}."
        )
        return Client(url=url, token=token, backend_type="remote")
    else:
        return Client(backend_type=backend_type)


def cancel_compute_plan(cp_id_path: str | Path):
    """
    Cancel a compute plan.

    We assume that we are in the remote setting.

    Parameters
    ----------
    cp_id_path : str or Path
        Path to the file containing the compute plan id.
        This file is a yaml file with the following structure:
        ```
        algo_org_name: str
        credentials_path: str
        compute_plan_key: str
        ```
    """
    try:
        with open(cp_id_path) as file:
            conf = yaml.load(file, Loader=yaml.FullLoader)

        algo_org_name = conf["algo_org_name"]
        credentials_path = conf["credentials_path"]
        client = get_client(
            backend_type="remote",
            org_name=algo_org_name,
            credentials_path=credentials_path,
        )
        compute_plan_key = conf["compute_plan_key"]
        client.cancel_compute_plan(compute_plan_key)
    except Exception as e:  # noqa : BLE001
        print(
            f"An error occured while cancelling the compute plan: {e}."
            f"Maybe it was already cancelled, or never launched ?"
        )


def get_n_centers_from_datasamples_file(datasamples_file: str | Path) -> int:
    """
    Return the number of centers from a datasamples file.

    Parameters
    ----------
    datasamples_file: str or Path
        Path to the yaml file containing the datasamples keys of the dataset.

    Returns
    -------
    int
        Number of centers in the datasamples file.

    """
    with open(datasamples_file) as file:
        dataset_datasamples_keys = yaml.load(file, Loader=yaml.FullLoader)
    return len(dataset_datasamples_keys)


def get_dependencies(
    backend_type: BackendType,
    fedpydeseq2_wheel_path: str | Path | None = None,
) -> Dependency:
    """
    Return a substra Dependency in regard to the backend_type.

    Parameters
    ----------
    backend_type : BackendType
        Name of the backend to connect to. Should be "subprocess", "docker" or "remote"
    fedpydeseq2_wheel_path : str | Path | None, optional
        Path to the wheel file of the fedpydeseq2 package. If provided and the backend
        is remote or docker, this wheel will be used instead of downloading it.

    Raises
    ------
    FileNotFoundError
        If the wheel file cannot be downloaded or found.
    """
    # in subprocess the dependency are not used, no need to build the wheel.
    if backend_type == BackendType.LOCAL_SUBPROCESS:
        return Dependency()

    if fedpydeseq2_wheel_path:
        wheel_path = Path(fedpydeseq2_wheel_path)
        if not wheel_path.exists():
            raise FileNotFoundError(f"Provided wheel file not found: {wheel_path}")
        logger.info(f"Using provided wheel path: {wheel_path}")
        return Dependency(local_installable_dependencies=[wheel_path])
    else:
        raise FileNotFoundError(
            "You must provide a wheel path when using a remote backend."
        )


def check_datasample_folder(datasample_folder: Path) -> None:
    """
    Sanity check for the datasample folder.

    Check if the datasample folder contains only two csv files: counts_data.csv
    and metadata.csv and nothing else.

    Parameters
    ----------
    datasample_folder : Path
        Path to the datasample folder.

    Raises
    ------
    ValueError
        If the datasample folder does not contain exactly two files named
        'counts_data.csv' and 'metadata.csv'.

    """
    if not datasample_folder.is_dir():
        raise ValueError(f"{datasample_folder} is not a directory.")
    files = list(datasample_folder.iterdir())
    if len(files) != 2:
        raise ValueError(
            "Datasample folder should contain exactly two files, "
            f"found {len(files)}: {files}."
        )
    if {file.name for file in files} != {"counts_data.csv", "metadata.csv"}:
        raise ValueError(
            "Datasample folder should contain two csv files named 'counts_data.csv'"
            " and 'metadata.csv'."
        )

    return
