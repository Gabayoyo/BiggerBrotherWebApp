import numpy as np

from dto.rep_metric import RepMetric


def estimate_rir_from_curve(
    current_set_reps: list[RepMetric], model: np.poly1d
) -> float:

    n = len(current_set_reps)
    if n == 0:
        raise ValueError("Need at least one rep of velocity data.")

    # Extract mean concentric velocities from each DTO
    v = np.array(
        [rep.mean_concentric_speed_ms for rep in current_set_reps], dtype=float
    )

    v_ref = v.max()
    vl_pct = max((v_ref - v[-1]) / v_ref * 100.0, 0.0)

    predicted_rep_pct = float(model(vl_pct))
    predicted_rep_pct = min(max(predicted_rep_pct, 100.0 / 30), 100.0)

    estimated_total = max(n / (predicted_rep_pct / 30), n)  # 30 = max plausible reps
    rir = max(round(estimated_total) - n, 0)

    return rir
