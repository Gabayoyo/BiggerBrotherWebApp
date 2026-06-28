import matplotlib.pyplot as plt
import numpy as np

from dto.rep_metric import RepMetric


def compute_vl_curve(
    prev_metrics: list[RepMetric], visualise_curve: bool = False
) -> np.poly1d:

    velocities = [
        m.mean_concentric_speed_ms
        for m in prev_metrics
        if m.mean_concentric_speed_ms is not None
    ]

    v = np.asarray(velocities, dtype=float)
    n = len(v)
    if n < 3:
        raise ValueError(f"Set has only {n} rep(s); need at least 3.")

    peak_idx = int(np.argmax(v))  # 0-based index of the fastest rep
    peak_position_pct = (peak_idx + 1) / n * 100.0

    # arbitrary threshold; if peak doesn't occur in the first third of the set, we don't have enough post-peak reps to characterize fatigue
    if peak_position_pct > 33.4:
        raise ValueError(
            f"Peak velocity occurs at rep {peak_idx + 1} of {n} "
            f"({peak_position_pct:.0f}% through the set), past the "
            f"{33.4:.0f}% threshold - only "
            f"{n - peak_idx} post-peak rep(s) remain, too few to "
            f"characterize a fatigue trend."
        )

    # Keep only the peak rep onward - the genuine fatigue trajectory.
    # Rep numbering for rep_pct stays anchored to the set's true total;
    # only the training pairs themselves are trimmed.
    v_post = v[peak_idx:]
    v_ref = v_post.max()
    vl_pct = (v_ref - v_post) / v_ref * 100.0
    rep_numbers = np.arange(peak_idx + 1, n + 1)
    rep_pct = rep_numbers / n * 100.0

    coeffs = np.polyfit(vl_pct, rep_pct, deg=2)

    if visualise_curve:
        plt.figure(figsize=(6, 5))
        plt.scatter(vl_pct, rep_pct, color="black", label="Calibration reps")
        x_smooth = np.linspace(vl_pct.min(), vl_pct.max(), 100)
        plt.plot(x_smooth, np.poly1d(coeffs)(x_smooth), "r-", label="Fitted VL curve")
        plt.xlabel("Velocity Loss (%)")
        plt.ylabel("Repetitions Completed (%)")
        plt.title("VL–%Repetitions Relationship")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return np.poly1d(coeffs)
