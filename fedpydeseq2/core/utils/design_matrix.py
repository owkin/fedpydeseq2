import warnings

import numpy as np
import pandas as pd
import pandas.api.types as ptypes


def build_design_matrix(
    metadata: pd.DataFrame,
    design_factors: str | list[str] = "stage",
    levels: dict[str, list[str]] | None = None,
    continuous_factors: list[str] | None = None,
    ref_levels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build design_matrix matrix for DEA.

    Unless specified, the reference factor is chosen alphabetically.
    Copied from PyDESeq2, with some modifications specific to fedomics to ensure that
    all centers have the same columns

    Parameters
    ----------
    metadata : pandas.DataFrame
        DataFrame containing metadata information.
        Must be indexed by sample barcodes.

    design_factors : str or list
        Name of the columns of metadata to be used as design_matrix variables.
        (default: ``"condition"``).

    levels : dict or None
        An optional dictionary of lists of strings specifying the levels of each factor
        in the global design, e.g. ``{"condition": ["A", "B"]}``. (default: ``None``).

    ref_levels : dict or None
        An optional dictionary of the form ``{"factor": "test_level"}``
        specifying for each factor the reference (control) level against which
        we're testing, e.g. ``{"condition", "A"}``. Factors that are left out
        will be assigned random reference levels. (default: ``None``).

    continuous_factors : list or None
        An optional list of continuous (as opposed to categorical) factors, that should
        also be in ``design_factors``. Any factor in ``design_factors`` but not in
        ``continuous_factors`` will be considered categorical (default: ``None``).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with experiment design information (to split cohorts).
        Indexed by sample barcodes.
    """
    if isinstance(
        design_factors, str
    ):  # if there is a single factor, convert to singleton list
        design_factors = [design_factors]

    # Check that factors in the design don't contain underscores. If so, convert
    # them to hyphens
    if np.any(["_" in factor for factor in design_factors]):
        warnings.warn(
            """Same factor names in the design contain underscores ('_'). They will
            be converted to hyphens ('-').""",
            UserWarning,
            stacklevel=2,
        )
        design_factors = [factor.replace("_", "-") for factor in design_factors]

    # Check that level factors in the design don't contain underscores. If so, convert
    # them to hyphens
    warning_issued = False
    for factor in design_factors:
        if ptypes.is_numeric_dtype(metadata[factor]):
            continue
        if np.any(["_" in value for value in metadata[factor]]):
            if not warning_issued:
                warnings.warn(
                    """Some factor levels in the design contain underscores ('_').
                    They will be converted to hyphens ('-').""",
                    UserWarning,
                    stacklevel=2,
                )
                warning_issued = True
            metadata[factor] = metadata[factor].apply(lambda x: x.replace("_", "-"))

    if continuous_factors is not None:
        for factor in continuous_factors:
            if factor not in design_factors:
                raise ValueError(
                    f"Continuous factor '{factor}' not in design factors: "
                    f"{design_factors}."
                )
        categorical_factors = [
            factor for factor in design_factors if factor not in continuous_factors
        ]
    else:
        categorical_factors = design_factors

    if levels is None:
        levels = {factor: np.unique(metadata[factor]) for factor in categorical_factors}

    # Check that there is at least one categorical factor
    if len(categorical_factors) > 0:
        design_matrix = pd.get_dummies(metadata[categorical_factors], drop_first=False)
        # Check if there missing levels. If so, add them and set to 0.
        for factor in categorical_factors:
            for level in levels[factor]:
                if f"{factor}_{level}" not in design_matrix.columns:
                    design_matrix[f"{factor}_{level}"] = 0

        # Pick the first level as reference. Then, drop the column.
        for factor in categorical_factors:
            if ref_levels is not None and factor in ref_levels:
                ref = ref_levels[factor]
            else:
                ref = levels[factor][0]

            ref_level_name = f"{factor}_{ref}"
            design_matrix.drop(ref_level_name, axis="columns", inplace=True)

            # Add reference level as column name suffix
            design_matrix.columns = [
                f"{col}_vs_{ref}" if col.startswith(factor) else col
                for col in design_matrix.columns
            ]
    else:
        # There is no categorical factor in the design
        design_matrix = pd.DataFrame(index=metadata.index)

    # Add the intercept column
    design_matrix.insert(0, "intercept", 1)

    # Convert categorical factors one-hot encodings to int
    design_matrix = design_matrix.astype("int")

    # Add continuous factors
    if continuous_factors is not None:
        for factor in continuous_factors:
            # This factor should be numeric
            design_matrix[factor] = pd.to_numeric(metadata[factor])
    return design_matrix
