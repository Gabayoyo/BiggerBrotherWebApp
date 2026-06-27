import sys
import os
import math
import numpy as np
import pytest

np.random.seed(0)

from dto.frame_data import Landmark, FrameData
from utils.utils import (
    smooth_floats,
    smooth_landmark_trajectory,
    sanitise_exercise_input,
    sanitise_unilateral_input,
)

def make_frame_data_list(positions, landmark_index=0):
    """Create a list of FrameData objects where the given landmark_index
    has the specified (x,y,z) positions. All other landmarks are zeros."""
    frames = []
    for pos in positions:
        landmarks = [Landmark(0.0, 0.0, 0.0, 1.0, 1.0) for _ in range(33)]
        landmarks[landmark_index] = Landmark(
            x=pos[0], y=pos[1], z=pos[2], visibility=1.0, presence=1.0
        )
        frames.append(FrameData(0, 0.0, landmarks, landmarks))
    return frames

def test_smooth_floats_basic():
    """A clean signal is smoothed without NaNs."""
    x = np.linspace(0, 10, 20)
    result = smooth_floats(x, window_size=5)
    assert isinstance(result, list)
    assert len(result) == len(x)
    assert not any(math.isnan(v) for v in result)
    # Smoothed values should be close to original (gentle slope)
    assert np.allclose(result, x, atol=0.5)


def test_smooth_floats_with_nans():
    """Internal NaNs are interpolated before smoothing."""
    signal = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    out = smooth_floats(signal, window_size=3)
    assert len(out) == len(signal)
    assert not any(np.isnan(v) for v in out)
    # The interpolated value should be ~3.0
    assert pytest.approx(out[2], 0.2) == 3.0


def test_smooth_floats_all_nans():
    """If all values are NaN, an exception is raised (empty non-NaN array)."""
    signal = np.full(10, np.nan)
    # Actual error from np.interp when all values are NaN
    with pytest.raises(ValueError, match="array of sample points is empty"):
        smooth_floats(signal, window_size=3)


def test_smooth_floats_window_too_large():
    """Window length must not exceed signal length."""
    signal = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="window_length must be less than or equal to the size of x"):
        smooth_floats(signal, window_size=5)


def test_smooth_floats_even_window():
    """
    Even window length is accepted by the current scipy version (≥ 1.8)
    but may produce unreliable smoothing. The function does not crash.
    """
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Should run without raising an exception
    out = smooth_floats(signal, window_size=4)
    assert len(out) == len(signal)

def test_smooth_landmark_trajectory_constant():
    """Constant positions stay constant after smoothing."""
    positions = np.array([[1.0, 2.0, 3.0]] * 5)
    frames = make_frame_data_list(positions, landmark_index=0)
    result = smooth_landmark_trajectory(frames, 0, window=5)
    assert result.shape == (5, 3)
    assert np.allclose(result, positions, atol=1e-6)


def test_smooth_landmark_trajectory_linear():
    """A linear trajectory is preserved (savgol on a straight line is exact)."""
    t = np.linspace(0, 1, 7)
    positions = np.array([[2 * ti, 3 * ti, -ti] for ti in t])
    frames = make_frame_data_list(positions, landmark_index=5)
    result = smooth_landmark_trajectory(frames, 5, window=5)
    assert result.shape == (7, 3)
    assert np.allclose(result, positions, atol=1e-6)


def test_smooth_landmark_trajectory_empty():
    """Empty frame data list raises a scipy error (window > data size)."""
    with pytest.raises(ValueError, match="window_length must be less than or equal"):
        smooth_landmark_trajectory([], 0)


def test_smooth_landmark_trajectory_bad_index():
    """Nonexistent landmark index raises IndexError."""
    frames = make_frame_data_list(np.zeros((3, 3)), 0)
    with pytest.raises(IndexError):
        smooth_landmark_trajectory(frames, 100)  # out of range

def test_sanitise_exercise_basic():
    assert sanitise_exercise_input("bicep_curl") == "bicep_curl"


def test_sanitise_exercise_whitespace_and_case():
    assert sanitise_exercise_input("  BICEP CURL  ") == "bicep_curl"


def test_sanitise_exercise_special_characters():
    assert sanitise_exercise_input("tricep-extension") == "tricep_extension"
    assert sanitise_exercise_input("lateral raise!") == "lateral_raise"


def test_sanitise_exercise_accents():
    assert sanitise_exercise_input("fiancée") == "fiancee"
    assert sanitise_exercise_input("Übung") == "ubung"


def test_sanitise_exercise_leading_digit_raises():
    with pytest.raises(ValueError, match="Must start with a letter"):
        sanitise_exercise_input("123abc")


def test_sanitise_exercise_only_digits_raises():
    with pytest.raises(ValueError, match="Must start with a letter"):
        sanitise_exercise_input("456")


def test_sanitise_exercise_empty_string_raises():
    with pytest.raises((IndexError, ValueError)):
        sanitise_exercise_input("")


def test_sanitise_exercise_multiple_spaces():
    assert sanitise_exercise_input("  chest   press  ") == "chest_press"


def test_sanitise_exercise_leading_trailing_underscores_removed():
    # Ensure underscores are stripped after substitution
    assert sanitise_exercise_input("__bench_press__") == "bench_press"

def test_sanitise_unilateral_valid():
    assert sanitise_unilateral_input("left") == "left"
    assert sanitise_unilateral_input("Right") == "right"
    assert sanitise_unilateral_input(" BILATERAL ") == "bilateral"


def test_sanitise_unilateral_invalid_raises():
    with pytest.raises(ValueError, match="Invalid unilateral value"):
        sanitise_unilateral_input("both")


def test_sanitise_unilateral_empty_raises():
    with pytest.raises(ValueError, match="Invalid unilateral value"):
        sanitise_unilateral_input("")