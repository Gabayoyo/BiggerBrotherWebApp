import pytest

from landmark_dicts import (
    ANGLE,
    ANGLES_OF_INTEREST,
    LANDMARK_INDICES,
    get_landmark_indices_from_angle,
    get_landmark_indices_from_exercise,
)


def test_get_landmark_indices_from_angle_valid():
    """Valid angle names return correct 3-tuple of indices."""
    # ELBOW_LEFT → LEFT_SHOULDER (11), LEFT_ELBOW (13), LEFT_WRIST (15)
    assert get_landmark_indices_from_angle("ELBOW_LEFT") == (11, 13, 15)

    # ELBOW_RIGHT
    assert get_landmark_indices_from_angle("ELBOW_RIGHT") == (12, 14, 16)

    # SHOULDER_LEFT
    assert get_landmark_indices_from_angle("SHOULDER_LEFT") == (23, 11, 13)

    # SHOULDER_RIGHT
    assert get_landmark_indices_from_angle("SHOULDER_RIGHT") == (24, 12, 14)

    # HIP_LEFT
    assert get_landmark_indices_from_angle("HIP_LEFT") == (11, 23, 25)

    # HIP_RIGHT
    assert get_landmark_indices_from_angle("HIP_RIGHT") == (12, 24, 26)


def test_get_landmark_indices_from_angle_all_defined():
    """Every key in ANGLE should return a tuple of valid indices."""
    for angle_name in ANGLE:
        indices = get_landmark_indices_from_angle(angle_name)
        assert isinstance(indices, tuple)
        assert len(indices) == 3
        for idx in indices:
            assert 0 <= idx <= 32  # valid MediaPipe index range


def test_get_landmark_indices_from_angle_invalid():
    """Invalid angle name raises ValueError."""
    with pytest.raises(ValueError, match="not defined in ANGLE"):
        get_landmark_indices_from_angle("NONEXISTENT")

    with pytest.raises(ValueError, match="not defined in ANGLE"):
        get_landmark_indices_from_angle("ELBOW")

    with pytest.raises(ValueError, match="not defined in ANGLE"):
        get_landmark_indices_from_angle("")


def test_get_landmark_indices_from_exercise_valid():
    """Valid exercises return list of angle tuples."""
    tricep = get_landmark_indices_from_exercise("tricep_extension")
    assert isinstance(tricep, list)
    assert len(tricep) == 2  # two angles: left and right elbow
    assert tricep[0] == (11, 13, 15)  # ELBOW_LEFT
    assert tricep[1] == (12, 14, 16)  # ELBOW_RIGHT

    bicep = get_landmark_indices_from_exercise("bicep_curl")
    assert isinstance(bicep, list)
    assert len(bicep) == 2
    assert bicep[0] == (11, 13, 15)
    assert bicep[1] == (12, 14, 16)


def test_get_landmark_indices_from_exercise_all_defined():
    """Every key in ANGLES_OF_INTEREST should return a non-empty list of valid tuples."""
    for ex_name in ANGLES_OF_INTEREST:
        tuples = get_landmark_indices_from_exercise(ex_name)
        assert isinstance(tuples, list)
        assert len(tuples) > 0
        for t in tuples:
            assert isinstance(t, tuple)
            assert len(t) == 3
            for idx in t:
                assert 0 <= idx <= 32


def test_get_landmark_indices_from_exercise_invalid():
    """Invalid exercise name raises ValueError."""
    with pytest.raises(ValueError, match="not defined in ANGLES_OF_INTEREST"):
        get_landmark_indices_from_exercise("running")

    with pytest.raises(ValueError, match="not defined in ANGLES_OF_INTEREST"):
        get_landmark_indices_from_exercise("")

    with pytest.raises(ValueError, match="not defined in ANGLES_OF_INTEREST"):
        get_landmark_indices_from_exercise("BICEP_CURL")  # exact string, case-sensitive


def test_exercise_angles_exist_in_angle_dict():
    """Every angle referenced by an exercise must exist in ANGLE."""
    for ex_name, angle_names in ANGLES_OF_INTEREST.items():
        for angle_name in angle_names:
            assert angle_name in ANGLE, (
                f"{angle_name} used by {ex_name} but not in ANGLE"
            )


def test_angle_landmark_names_exist_in_indices():
    """Every landmark name used in ANGLE must exist in LANDMARK_INDICES."""
    for angle_name, landmark_names in ANGLE.items():
        for lm_name in landmark_names:
            assert lm_name in LANDMARK_INDICES, (
                f"{lm_name} in {angle_name} missing from LANDMARK_INDICES"
            )
