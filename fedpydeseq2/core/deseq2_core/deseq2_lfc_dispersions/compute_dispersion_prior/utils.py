from typing import Union

import numpy as np
import pandas as pd
from pydeseq2.utils import dispersion_trend


def disp_function(
    x,
    disp_function_type,
    coeffs: Union["pd.Series[float]", np.ndarray] | None = None,
    mean_disp: float | None = None,
) -> float | np.ndarray:
    """Return the dispersion trend function at x."""
    if disp_function_type == "parametric":
        assert coeffs is not None, "coeffs must be provided for parametric dispersion."
        return dispersion_trend(x, coeffs=coeffs)
    elif disp_function_type == "mean":
        assert mean_disp is not None, "mean_disp must be provided for mean dispersion."
        return np.full_like(x, mean_disp)
    else:
        raise ValueError(
            "disp_function_type must be 'parametric' or 'mean',"
            f" got {disp_function_type}"
        )
