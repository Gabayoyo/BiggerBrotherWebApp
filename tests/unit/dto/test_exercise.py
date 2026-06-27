import sys
import os
import numpy as np
import pytest

np.random.seed(0)

from dto.exercise import exercise, get_exercise
from dto.frame_data import FrameData, Landmark
from landmark_dicts import get_landmark_indices_from_exercise


@pytest.fixture
def frame_data_list(make_landmarks):
    """Returns a list containing one FrameData for convenience."""
    lms = make_landmarks()
    return [FrameData(frame_number=0, timestamp_s=0.0, landmarks=lms, world_landmarks=lms)] 

def test_exercise_direct_creation():
    """An exercise instance stores attributes exactly as given."""
    # Simple limb tuple (left arm only for a mock exercise)
    limbs = [(11, 13, 15)]
    ex = exercise(
        name="test_exercise",
        limbs=limbs,
        bilateral="left",
        is_flexion=True,
        frame_data=None
    )
    assert ex.name == "test_exercise"
    assert ex.limbs == limbs
    assert ex.bilateral == "left"
    assert ex.is_flexion is True
    assert ex.frame_data is None


def test_exercise_with_frame_data(frame_data_list):
    """The frame_data list is stored as provided."""
    ex = exercise(
        name="bicep_curl",
        limbs=[(11, 13, 15), (12, 14, 16)],
        bilateral="bilateral",
        is_flexion=True,
        frame_data=frame_data_list
    )
    assert ex.frame_data is frame_data_list
    assert len(ex.frame_data) == 1

def test_get_exercise_bilateral_bicep_curl(frame_data_list):
    """Bilateral bicep_curl returns both limb tuples and correct flags."""
    ex = get_exercise("bicep_curl", "bilateral", frame_data_list)

    assert ex.name == "bicep_curl"
    assert ex.bilateral == "bilateral"
    assert ex.is_flexion is True
    assert ex.frame_data is frame_data_list

    # limbs should be a list of two tuples (left + right arm)
    expected_limbs = get_landmark_indices_from_exercise("bicep_curl")
    assert len(expected_limbs) == 2
    assert ex.limbs == expected_limbs   # exactly as returned


def test_get_exercise_left_bicep_curl(frame_data_list):
    """Left‑only bicep_curl returns only the left limb tuple."""
    ex = get_exercise("bicep_curl", "left", frame_data_list)

    assert ex.bilateral == "left"
    # limbs must contain exactly one tuple, and it must be the first of the original pair
    full_limbs = get_landmark_indices_from_exercise("bicep_curl")
    assert len(ex.limbs) == 1
    assert ex.limbs[0] == full_limbs[0]   # left side is index 0


def test_get_exercise_right_bicep_curl(frame_data_list):
    """Right‑only bicep_curl returns only the right limb tuple."""
    ex = get_exercise("bicep_curl", "right", frame_data_list)

    assert ex.bilateral == "right"
    full_limbs = get_landmark_indices_from_exercise("bicep_curl")
    assert len(ex.limbs) == 1
    assert ex.limbs[0] == full_limbs[1]   # right side is index 1


def test_get_exercise_tricep_extension(frame_data_list):
    """Tricep extension has is_flexion=False."""
    ex = get_exercise("tricep_extension", "bilateral", frame_data_list)
    assert ex.is_flexion is False
    assert ex.name == "tricep_extension"


def test_get_exercise_unknown_name():
    """Unknown exercise names raise ValueError."""
    with pytest.raises(ValueError, match="Unknown exercise"):
        get_exercise("nonexistent", "bilateral", [])


def test_get_exercise_unilateral_not_possible(frame_data_list):
    """
    If an exercise does not support unilateral but a left/right setting
    is given, a ValueError should be raised.
    We temporarily add a fake exercise that has uni_possible=False.
    """
    from dto.exercise import _EXERCISE_CONFIGS

    original = _EXERCISE_CONFIGS.copy()
    _EXERCISE_CONFIGS["test_only_bilateral"] = {
        "uni_possible": False,
        "is_flexion": True
    }
    try:
        with pytest.raises(ValueError):
            get_exercise("test_only_bilateral", "left", frame_data_list)
    finally:
        _EXERCISE_CONFIGS.clear()
        _EXERCISE_CONFIGS.update(original)


def test_get_exercise_unilateral_not_possible_with_mock(frame_data_list, monkeypatch):
    """
    Proper test: mock get_landmark_indices_from_exercise to return dummy limbs,
    then expect ValueError because uni_possible is False.
    """
    from dto.exercise import _EXERCISE_CONFIGS

    # Add temporary config
    _EXERCISE_CONFIGS["test_uni"] = {"uni_possible": False, "is_flexion": True}
    monkeypatch.setattr(
        "dto.exercise.get_landmark_indices_from_exercise",
        lambda name: [(0, 1, 2), (3, 4, 5)]
    )

    with pytest.raises(ValueError, match="cannot be performed unilaterally"):
        get_exercise("test_uni", "left", frame_data_list)

    # Clean up
    del _EXERCISE_CONFIGS["test_uni"]


def test_get_exercise_bilateral_does_not_trigger_unilateral_check(frame_data_list, monkeypatch):
    """
    Even if uni_possible=False, using 'bilateral' should work.
    """
    from dto.exercise import _EXERCISE_CONFIGS

    _EXERCISE_CONFIGS["test_bilat_only"] = {"uni_possible": False, "is_flexion": False}
    monkeypatch.setattr(
        "dto.exercise.get_landmark_indices_from_exercise",
        lambda name: [(0, 1, 2), (3, 4, 5)]
    )

    # This should NOT raise
    ex = get_exercise("test_bilat_only", "bilateral", frame_data_list)
    assert ex.bilateral == "bilateral"
    assert ex.limbs == [(0, 1, 2), (3, 4, 5)]

    del _EXERCISE_CONFIGS["test_bilat_only"]


def test_get_exercise_empty_frame_data():
    """Passing an empty list for frame_data stores it as is."""
    ex = get_exercise("bicep_curl", "bilateral", [])
    assert ex.frame_data == []