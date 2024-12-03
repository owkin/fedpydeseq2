DEFAULT_PYDESEQ2_REF_LEVELS = {
    "stage": "Advanced",
    "gender": "female",
}


def make_reference_and_fl_ref_levels(
    design_factors: list[str],
    continuous_factors: list[str] | None = None,
    ref_levels: dict[str, str] | None = None,
    reference_dds_ref_level: tuple[str, str] | None = None,
) -> tuple[dict[str, str] | None, tuple[str, str] | None]:
    """Function to make reference levels for PyDESeq2 and FedPyDESeq2.

    The goal of this function is to enforce that the design matrices will be
    comparable between PyDESeq2 and FedPyDESeq2. This is done by ensuring that
    the reference levels are the same for both packages.

    Parameters
    ----------
    design_factors : list[str]
        List of factors in the design matrix.
    continuous_factors : list[str] or None
        List of continuous factors in the design matrix, by default None.
    ref_levels : dict[str, str] or None
        Reference levels for the factors in the design matrix, by default None.
        These are the reference levels used as arguments in the
        FedPyDESeq2Strategy class.
    reference_dds_ref_level : tuple[str, str] or None
        Reference level for the factor in the design matrix, by default None.
        This is the reference level used as an argument in the DESeqDataSet
        class from the pydeseq2 package.


    Returns
    -------
    complete_ref_levels : Optional[dict[str, str]]
        Reference levels for the factors in the design matrix for the
        FedPyDESeq2 package.
    reference_dds_ref_level : Optional[tuple[str, str]]
        Reference level for the factor in the design matrix for the
        PyDESeq2 package.

    """
    categorical_factors = (
        design_factors
        if continuous_factors is None
        else [factor for factor in design_factors if factor not in continuous_factors]
    )
    complete_ref_levels = {
        factor: level
        for factor, level in DEFAULT_PYDESEQ2_REF_LEVELS.items()
        if factor in categorical_factors
    }
    if ref_levels is not None:
        if len(ref_levels) > 1:
            print(
                "Warning: only one reference level is supported when comparing with "
                "PyDESeq2. The first reference level will be used."
            )
        ref_factor, ref_level = next(iter(ref_levels.items()))
        if reference_dds_ref_level is not None:
            assert ref_factor == reference_dds_ref_level[0]
            assert ref_level == reference_dds_ref_level[1]
        else:
            reference_dds_ref_level = (ref_factor, ref_level)
        complete_ref_levels[ref_factor] = ref_level
    elif reference_dds_ref_level is not None:
        ref_factor, ref_level = reference_dds_ref_level
        complete_ref_levels[ref_factor] = ref_level

    return complete_ref_levels, reference_dds_ref_level
