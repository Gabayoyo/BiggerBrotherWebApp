import numpy as np
from scipy.signal import savgol_filter
import re
import unicodedata

from dto.frame_data import FrameData
    
def smooth_floats(signal: np.array, window_size: int = 5) -> list[float]:
    """Apply a moving average filter to smooth the signal."""

    # Copy to avoid modifying the original array
    signal = signal.copy().astype(float)
    
    # Find indices where signal is not NaN
    not_nan = ~np.isnan(signal)
    if not np.all(not_nan):
        # Interpolate NaNs using linear interpolation (or more advanced methods)
        x = np.arange(len(signal))
        signal[np.isnan(signal)] = np.interp(
            x[np.isnan(signal)], x[not_nan], signal[not_nan]
        )

    smoothed = savgol_filter(signal, window_length=window_size, polyorder=2)
    return smoothed.tolist()

def smooth_landmark_trajectory(
    frame_data: list[FrameData],
    landmark_index: int,
    window: int = 5,
    polyorder: int = 2
) -> np.ndarray:
    """
    Extract and smooth a single landmark's (x, y, z) over time.
    Returns (n_frames, 3) array of smoothed coordinates.
    """
    positions = np.array([f.get_landmark(landmark_index).to_array() for f in frame_data])
    # Apply smoothing along time axis (axis=0)
    smoothed = savgol_filter(positions, window_length=window, polyorder=polyorder, axis=0)
    return smoothed

def sanitise_exercise_input(s: str) -> str:
    """
    Normalise any string into a Python‑friendly snake_case identifier.
    - Lowercase
    - Strip leading/trailing whitespace
    - Decompose accented characters (é → e, ñ → n)
    - Replace any run of non‑alphanumeric characters (spaces, hyphens, etc.)
      with a single underscore
    - Collapse multiple underscores
    - Remove leading/trailing underscores
    - Guarantee a non‑empty, non‑numeric‑first string (prepend 'x' if needed)
    """
    # 1. Strip and lowercase
    s = s.strip().lower()

    # 2. Unicode normalisation: split ligatures and accents (NFKD)
    #    ‘ﬁ’ → ‘fi’, ‘é’ → ‘e’ + combining accent
    s = unicodedata.normalize('NFKD', s)

    # 3. Remove diacritical marks (combining characters category ‘M’)
    s = ''.join(c for c in s if not unicodedata.category(c).startswith('M'))

    # 4. Replace any sequence of non‑alphanumeric characters (including spaces,
    #    hyphens, punctuation) with a single underscore.
    #    Keep letters a-z and digits 0-9, everything else becomes '_'.
    s = re.sub(r'[^a-z0-9]+', '_', s)

    # 5. Remove leading/trailing underscores
    s = s.strip('_')

    # 6. Ensure it starts with a letter or underscore (Python identifier rule)
    #    If empty or starts with a digit, prepend 'x_'
    if not s or s[0].isdigit():
        s = 'x_' + s

    # 7. Final deduplication of underscores (already handled by step 4, but safe)
    s = re.sub(r'_+', '_', s)

    return s

def sanitise_unilateral_input(s: str) -> str:
    s = s.strip().lower()
    if s in ["bilateral", "left", "right"]:
        return s
    else:
        raise ValueError("Invalid unilateral value. Must be 'bilateral', 'left', or 'right'.")