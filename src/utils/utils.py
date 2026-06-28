import re
import unicodedata

import numpy as np
from scipy.signal import savgol_filter

from dto.frame_data import FrameData


# used to smooth a 1D array of floats, e.g. angles, using a Savitzky-Golay filter
def smooth_floats(signal: np.array, window_size: int = 5) -> list[float]:

    signal = signal.copy().astype(float)

    # Find indices where signal is not NaN
    not_nan = ~np.isnan(signal)
    if not np.all(not_nan):
        # Interpolate NaNs using linear interpolation
        x = np.arange(len(signal))
        signal[np.isnan(signal)] = np.interp(
            x[np.isnan(signal)], x[not_nan], signal[not_nan]
        )

    smoothed = savgol_filter(signal, window_length=window_size, polyorder=2)
    return smoothed.tolist()


# used to smooth the trajectory of a single landmark's xyz position over time
# returns a (n_frames, 3) array of smoothed coordinates
def smooth_landmark_trajectory(
    frame_data: list[FrameData],
    landmark_index: int,
    window: int = 5,
    polyorder: int = 2,
) -> np.ndarray:
    positions = np.array(
        [f.get_landmark(landmark_index).to_array() for f in frame_data]
    )

    # apply smoothing along time axis (axis=0)
    smoothed = savgol_filter(
        positions, window_length=window, polyorder=polyorder, axis=0
    )
    return smoothed


# sanitises exercise name input to a consistent format for internal use
# e.g. TrIcep  Extension -> tricep_extension
def sanitise_exercise_input(s: str) -> str:

    # to lowercase
    s = s.strip().lower()

    # normalise unicode characters to their closest ASCII equivalent
    s = unicodedata.normalize("NFKD", s)

    # remove diacritical marks like accents etc.
    s = "".join(c for c in s if not unicodedata.category(c).startswith("M"))

    # replace spaces and non-alphanumeric characters with underscores
    s = re.sub(r"[^a-z0-9]+", "_", s)

    # remove leading/trailing underscores
    s = s.strip("_")

    # must start with a letter
    if not s[0].isalpha():
        raise ValueError("Invalid exercise name. Must start with a letter.")

    return s


# quick check for valid exercise name format
def sanitise_unilateral_input(s: str) -> str:
    s = s.strip().lower()
    if s in ["bilateral", "left", "right"]:
        return s
    else:
        raise ValueError(
            "Invalid unilateral value. Must be 'bilateral', 'left', or 'right'."
        )
