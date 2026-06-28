import numpy as np
from scipy.signal import butter, filtfilt
from numpy.typing import NDArray

from dto.exercise import Exercise
from landmark_dicts import LANDMARK_OF_INTEREST


# main function to derive velocity from frame data
def derive_velocity(
    exercise: Exercise,
    fps: float,
) -> list[float]:
    n_frames = len(exercise.frame_data)
    if n_frames < 2:
        return [0.0] * n_frames

    # Collect raw positions per frame as a list first
    raw_positions = []

    for frame in exercise.frame_data:
        # ... (the existing logic to compute `pos`) ...
        if exercise.bilateral == "right":
            lm_key_1 = LANDMARK_OF_INTEREST[exercise.name][1]
            pos_right, conf_right = frame.get_lm_xyzv_by_name(lm_key_1)
            pos = pos_right if conf_right > 0.7 else np.full(3, np.nan)
        else:
            lm_key_1 = LANDMARK_OF_INTEREST[exercise.name][0]
            pos_left, conf_left = frame.get_lm_xyzv_by_name(lm_key_1)

            if exercise.bilateral == "bilateral":
                lm_key_2 = LANDMARK_OF_INTEREST[exercise.name][1]
                pos_right, conf_right = frame.get_lm_xyzv_by_name(lm_key_2)
                total = conf_left + conf_right
                if total > 0:
                    pos = (pos_left * conf_left + pos_right * conf_right) / total
                else:
                    pos = np.full(3, np.nan)
            elif exercise.bilateral == "left" and conf_left > 0.7:
                pos = pos_left
            else:
                pos = np.full(3, np.nan)

        raw_positions.append(pos)

    # Convert to typed NumPy array
    positions_arr: NDArray[np.float64] = np.array(raw_positions, dtype=np.float64)

    # Interpolate NaNs per axis
    if np.any(np.isnan(positions_arr)):
        for col in range(3):
            col_vals = positions_arr[:, col]
            nans = np.isnan(col_vals)
            if np.any(nans):
                valid_idx = np.where(~nans)[0]
                if len(valid_idx) == 0:
                    positions_arr[:, col] = 0.0
                    continue
                col_vals[nans] = np.interp(
                    np.flatnonzero(nans), valid_idx, col_vals[valid_idx]
                )
                positions_arr[:, col] = col_vals

    # Low-pass filter each axis
    cutoff_hz = min(6.0, fps * 0.4)
    nyq = 0.5 * fps
    b, a = butter(4, cutoff_hz / nyq, btype="low", analog=False)
    for col in range(3):
        positions_arr[:, col] = filtfilt(b, a, positions_arr[:, col], axis=0)  # type: ignore[assignment]

    # Central-difference velocity
    dt = 1.0 / fps
    velocities = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        if i == 0:
            v = (positions_arr[1] - positions_arr[0]) / dt
        elif i == n_frames - 1:
            v = (positions_arr[-1] - positions_arr[-2]) / dt
        else:
            v = (positions_arr[i + 1] - positions_arr[i - 1]) / (2 * dt)
        velocities[i] = np.linalg.norm(v)

    return velocities.tolist()