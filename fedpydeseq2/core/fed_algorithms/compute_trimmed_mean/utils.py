def trimfn(x: float) -> int:
    """
    Determine the use-case of the trim ratio and scale based on cell counts.

    Parameters
    ----------
    x : float
        The number of cells.

    Returns
    -------
    int
        The index of the trim ratio and scale to use.
    """
    return 2 if x >= 23.5 else 1 if x >= 3.5 else 0


def get_trim_ratio(x):
    """
    Get the trim ratio based on the number of cells.

    Parameters
    ----------
    x : float
        The number of cells.

    Returns
    -------
    float
        The trim ratio.
    """
    trimratio = (1 / 3, 1 / 4, 1 / 8)
    return trimratio[trimfn(x)]


def get_scale(x):
    """
    Get the scale based on the number of cells.

    Parameters
    ----------
    x : float
        The number of cells.

    Returns
    -------
    float
        The scale used to compute the dispersion during cook distance calculation.

    """
    scales = (2.04, 1.86, 1.51)
    return scales[trimfn(x)]
