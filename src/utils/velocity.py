import numpy as np
from scipy.signal import butter, filtfilt
from dto.exercise import exercise
from dto.frame_data import FrameData
from landmark_dicts import LANDMARK_OF_INTEREST

def derive_velocity(
    exercise: exercise,
    fps: float,
):
    n_frames = len(exercise.frame_data)
    if n_frames < 2:
        return [0.0] * n_frames

    # LANDMARK_OF_INTEREST returns (left_landmark_index, right_landmark_index)

    # --- 1. Extract landmark positions ---
    positions = []

    for frame in exercise.frame_data:

        if exercise.bilateral == "right":
            lm_key_1 = LANDMARK_OF_INTEREST[exercise.name][1]
            pos_right, conf_right = frame.get_lm_xyzv_by_name(lm_key_1)
            if conf_right > 0.7:
                pos = pos_right
            else:
                pos = np.full(3, np.nan)
        else:
            lm_key_1 = LANDMARK_OF_INTEREST[exercise.name][0]
            pos_left, conf_left = frame.get_lm_xyzv_by_name(lm_key_1)

            if exercise.bilateral == "bilateral":
                lm_key_2 = LANDMARK_OF_INTEREST[exercise.name][1]
                pos_right, conf_right = frame.get_lm_xyzv_by_name(lm_key_2)

                total = conf_left + conf_right
                pos = (pos_left * conf_left + pos_right * conf_right) / total

            elif exercise.bilateral == "left" and conf_left > 0.7:
                pos = pos_left
            else:
                pos = np.full(3, np.nan)

        positions.append(pos)

    positions = np.array(positions, dtype=np.float64)  # shape (N, 3)

    # --- 2. Handle missing data (NaNs from invisible frames) ---
    if np.any(np.isnan(positions)):
        for col in range(3):
            col_vals = positions[:, col]
            nans = np.isnan(col_vals)
            if np.any(nans):
                valid_idx = np.where(~nans)[0]
                if len(valid_idx) == 0:
                    # Every frame invisible on this axis — zero out
                    positions[:, col] = 0.0
                    continue
                col_vals[nans] = np.interp(
                    np.flatnonzero(nans), valid_idx, col_vals[valid_idx]
                )
                positions[:, col] = col_vals

    # --- 3. Low-pass filter ---
    nyq = 0.5 * fps
    b, a = butter(4, 6 / nyq, btype='low', analog=False)
    for col in range(3):
        positions[:, col] = filtfilt(b, a, positions[:, col], axis=0)

    # --- 4. Compute velocity magnitude using central differences ---
    dt = 1.0 / fps
    velocities = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        if i == 0:
            v = (positions[1] - positions[0]) / dt
        elif i == n_frames - 1:
            v = (positions[-1] - positions[-2]) / dt
        else:
            v = (positions[i + 1] - positions[i - 1]) / (2 * dt)
        velocities[i] = np.linalg.norm(v)

    return velocities.tolist()