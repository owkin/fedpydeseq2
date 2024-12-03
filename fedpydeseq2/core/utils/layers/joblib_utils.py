from typing import Any


def get_joblib_parameters(x: Any) -> tuple[int, int, str, int]:
    """
    Get the joblib parameters from an object, and return them as a tuple.

    If the object has no joblib parameters, default values are returned.

    Parameters
    ----------
    x: Any
        Object from which to extract the joblib parameters.

    Returns
    -------
    n_jobs: int
        Number of jobs to run in parallel.
    joblib_verbosity: int
        Verbosity level of joblib.
    joblib_backend: str
        Joblib backend.
    batch_size: int
        Batch size for the IRLS algorithm.

    """
    n_jobs = x.num_jobs if hasattr(x, "num_jobs") else 1

    joblib_verbosity = x.joblib_verbosity if hasattr(x, "joblib_verbosity") else 0
    joblib_backend = x.joblib_backend if hasattr(x, "joblib_backend") else "loky"
    batch_size = x.irls_batch_size if hasattr(x, "irls_batch_size") else 100
    return n_jobs, joblib_verbosity, joblib_backend, batch_size
