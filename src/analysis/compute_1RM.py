import math


def compute_1rm(weight: float, reps: int) -> float:
    """
    Calculate one‑rep max (1rm) using the formula:
        1rm = weight * (1 + (reps - 1) ** 0.85 / (-2.55 + 4.58 * ln(weight)))

    Parameters
    ----------
    weight : float
        Weight lifted (must be > 0).
    reps : int
        Number of repetitions performed (must be >= 1).

    Returns
    -------
    float
        Estimated one‑rep max.

    Raises
    ------
    ValueError
        If weight <= 0, reps < 1, or the denominator of the formula
        is zero (which occurs when weight ≈ 1.744).
    """
    if weight <= 0:
        raise ValueError("Weight must be greater than 0.")
    if reps < 1:
        raise ValueError("Reps must be at least 1.")

    # Denominator: -2.55 + 4.58 * ln(weight)
    denom = -2.55 + 4.58 * math.log(weight)

    # Avoid division by zero (happens when weight = e^(2.55/4.58) ≈ 1.744)
    if denom == 0:
        raise ValueError(
            "Denominator is zero for this weight. "
            "Choose a weight different from approximately 1.744."
        )

    one_rm = weight * (1 + (reps - 1) ** 0.85 / denom)
    return one_rm
