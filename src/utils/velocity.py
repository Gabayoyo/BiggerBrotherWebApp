import numpy as np
from scipy.signal import butter, filtfilt

from dto.exercise import Exercise
from landmark_dicts import LANDMARK_OF_INTEREST


# main function to derive velocity from frame data
def derive_velocity(
    exercise: Exercise,
    fps: float,
):
    n_frames = len(exercise.frame_data)
    if n_frames < 2:
        return [0.0] * n_frames

    # get landmark positions for the exercise of interest, handling bilateral cases
    positions = []

    # iterate every frame, get frame's xyz position, average it if bilateral
    # then apply a low-pass filter, then compute velocity using central differences
    for frame in exercise.frame_data:
        # if just right side, use right landmark (always index 1 for LANDMARK_OF_INTEREST)
        if exercise.bilateral == "right":
            lm_key_1 = LANDMARK_OF_INTEREST[exercise.name][1]
            pos_right, conf_right = frame.get_lm_xyzv_by_name(lm_key_1)
            pos = pos_right if conf_right > 0.7 else np.full(3, np.nan)
        else:
            # else at the very least is left
            lm_key_1 = LANDMARK_OF_INTEREST[exercise.name][0]
            pos_left, conf_left = frame.get_lm_xyzv_by_name(lm_key_1)

            # check if bilateral, then get right side too and average based on confidence
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

        positions.append(pos)

    positions = np.array(positions, dtype=np.float64)  # shape (N, 3)

    # handle nans by interpolating missing values for each axis separately
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

    # apply low pass filter to smooth the positions
    cutoff_hz = min(6.0, fps * 0.4)
    nyq = 0.5 * fps
    b, a = butter(4, cutoff_hz / nyq, btype="low", analog=False)
    for col in range(3):
        positions[:, col] = filtfilt(b, a, positions[:, col], axis=0)

    # compute velocity using central differences
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
