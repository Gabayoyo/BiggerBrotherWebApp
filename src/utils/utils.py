import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

from dto.frame_data import FrameData
    
def smooth_floats(signal: np.array, window_size: int = 5) -> list[float]:
    """Apply a moving average filter to smooth the signal."""
    result = savgol_filter(signal, window_length=window_size, polyorder=2).tolist()
    return result

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