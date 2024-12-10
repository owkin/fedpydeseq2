from typing import Literal

import numpy as np
from scipy.stats import norm  # type: ignore


def build_contrast(
    design_factors,
    design_columns,
    continuous_factors=None,
    contrast: list[str] | None = None,
) -> list[str]:
    """Check the validity of the contrast (if provided).

    If not, build a default
    contrast, corresponding to the last column of the design matrix.
    A contrast should be a list of three strings, in the following format:
    ``['variable_of_interest', 'tested_level', 'reference_level']``.
    Names must correspond to the metadata data passed to the FedCenters.
    E.g., ``['condition', 'B', 'A']`` will measure the LFC of 'condition B'
    compared to 'condition A'.
    For continuous variables, the last two strings will be left empty, e.g.
    ``['measurement', '', ''].
    If None, the last variable from the design matrix
    is chosen as the variable of interest, and the reference level is picked
    alphabetically.

    Parameters
    ----------
    design_factors : list
        The design factors.
    design_columns : list
        The names of the columns of the design matrices in the centers.
    continuous_factors : list, optional
        The continuous factors in the design, if any. (default: ``None``).
    contrast : list, optional
        A list of three strings, in the following format:
        ``['variable_of_interest', 'tested_level', 'reference_level']``.
        (default: ``None``).
    """
    if contrast is not None:  # Test contrast if provided
        if len(contrast) != 3:
            raise ValueError("The contrast should contain three strings.")
        if contrast[0] not in design_factors:
            raise KeyError(
                f"The contrast variable ('{contrast[0]}') should be one "
                f"of the design factors."
            )
        # TODO: Ideally, we should check that the levels are valid. This might leak
        # data from the centers, though.

    else:  # Build contrast if None
        factor = design_factors[-1]
        # Check whether this factor is categorical or continuous.
        if continuous_factors is not None and factor in continuous_factors:
            # The factor is continuous
            contrast = [factor, "", ""]
        else:
            # The factor is categorical
            factor_col = next(col for col in design_columns if col.startswith(factor))
            split_col = factor_col.split("_")
            contrast = [split_col[0], split_col[1], split_col[-1]]

    return contrast


def build_contrast_vector(contrast, LFC_columns) -> tuple[np.ndarray, int | None]:
    """
    Build a vector corresponding to the desired contrast.

    Allows to test any pair of levels without refitting LFCs.

    Parameters
    ----------
    contrast : list
        A list of three strings, in the following format:
        ``['variable_of_interest', 'tested_level', 'reference_level']``.
    LFC_columns : list
        The names of the columns of the LFC matrices in the centers.

    Returns
    -------
    contrast_vector : ndarray
        The contrast vector, containing multipliers to apply to the LFCs.
    contrast_idx : int, optional
        The index of the tested contrast in the LFC matrix.
    """
    factor = contrast[0]
    alternative = contrast[1]
    ref = contrast[2]
    if ref == alternative == "":
        # "factor" is a continuous variable
        contrast_level = factor
    else:
        contrast_level = f"{factor}_{alternative}_vs_{ref}"

    contrast_vector = np.zeros(len(LFC_columns))
    if contrast_level in LFC_columns:
        contrast_idx = LFC_columns.get_loc(contrast_level)
        contrast_vector[contrast_idx] = 1
    elif f"{factor}_{ref}_vs_{alternative}" in LFC_columns:
        # Reference and alternative are inverted
        contrast_idx = LFC_columns.get_loc(f"{factor}_{ref}_vs_{alternative}")
        contrast_vector[contrast_idx] = -1
    else:
        # Need to change reference
        # Get any column corresponding to the desired factor and extract old ref
        old_ref = next(col for col in LFC_columns if col.startswith(factor)).split(
            "_vs_"
        )[-1]
        new_alternative_idx = LFC_columns.get_loc(
            f"{factor}_{alternative}_vs_{old_ref}"
        )
        new_ref_idx = LFC_columns.get_loc(f"{factor}_{ref}_vs_{old_ref}")
        contrast_vector[new_alternative_idx] = 1
        contrast_vector[new_ref_idx] = -1
        # In that case there is no contrast index
        contrast_idx = None

    return contrast_vector, contrast_idx


def wald_test(
    M: np.ndarray,
    lfc: np.ndarray,
    ridge_factor: np.ndarray | None,
    contrast_vector: np.ndarray,
    lfc_null: float,
    alt_hypothesis: Literal["greaterAbs", "lessAbs", "greater", "less"] | None,
) -> tuple[float, float, float]:
    """Run Wald test for a single gene.

    Computes Wald statistics, standard error and p-values from
    dispersion and LFC estimates.

    Parameters
    ----------
    M : ndarray
        Central parameter in the covariance matrix estimator.

    lfc : ndarray
        Log-fold change estimate (in natural log scale).

    ridge_factor : ndarray, optional
        Regularization factors.

    contrast_vector : ndarray
        Vector encoding the contrast that is being tested.

    lfc_null : float
        The log fold change (in natural log scale) under the null hypothesis.

    alt_hypothesis : str, optional
        The alternative hypothesis for computing wald p-values.

    Returns
    -------
    wald_p_value : float
        Estimated p-value.

    wald_statistic : float
        Wald statistic.

    wald_se : float
        Standard error of the Wald statistic.
    """
    # Build covariance matrix estimator

    if ridge_factor is None:
        ridge_factor = np.diag(np.repeat(1e-6, M.shape[0]))
    H = np.linalg.inv(M + ridge_factor)
    Hc = H @ contrast_vector
    # Evaluate standard error and Wald statistic
    wald_se: float = np.sqrt(Hc.T @ M @ Hc)

    def greater(lfc_null):
        stat = contrast_vector @ np.fmax((lfc - lfc_null) / wald_se, 0)
        pval = norm.sf(stat)
        return stat, pval

    def less(lfc_null):
        stat = contrast_vector @ np.fmin((lfc - lfc_null) / wald_se, 0)
        pval = norm.sf(np.abs(stat))
        return stat, pval

    def greater_abs(lfc_null):
        stat = contrast_vector @ (
            np.sign(lfc) * np.fmax((np.abs(lfc) - lfc_null) / wald_se, 0)
        )
        pval = 2 * norm.sf(np.abs(stat))  # Only case where the test is two-tailed
        return stat, pval

    def less_abs(lfc_null):
        stat_above, pval_above = greater(-abs(lfc_null))
        stat_below, pval_below = less(abs(lfc_null))
        return min(stat_above, stat_below, key=abs), max(pval_above, pval_below)

    wald_statistic: float
    wald_p_value: float
    if alt_hypothesis:
        wald_statistic, wald_p_value = {
            "greaterAbs": greater_abs(lfc_null),
            "lessAbs": less_abs(lfc_null),
            "greater": greater(lfc_null),
            "less": less(lfc_null),
        }[alt_hypothesis]
    else:
        wald_statistic = contrast_vector @ (lfc - lfc_null) / wald_se
        wald_p_value = 2 * norm.sf(np.abs(wald_statistic))

    return wald_p_value, wald_statistic, wald_se
