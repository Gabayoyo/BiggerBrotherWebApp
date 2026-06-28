import sys
import os
import numpy as np
import pytest

np.random.seed(0)

from dto.frame_data import Landmark, FrameData
from dto.exercise import exercise
from utils.angle import _calculate_angle, derive_angles
from landmark_dicts import get_landmark_indices_from_exercise

def make_limb_landmarks(
    shoulder_xyz,
    elbow_xyz,
    wrist_xyz,
    vis_shld=1.0, vis_elb=1.0, vis_wrs=1.0
):
    """
    Returns a list of three Landmark objects representing a limb.
    Coordinates are in world space as expected by get_limb (world_landmarks).
    """
    return [
        Landmark(x=shoulder_xyz[0], y=shoulder_xyz[1], z=shoulder_xyz[2],
                 visibility=vis_shld, presence=1.0),
        Landmark(x=elbow_xyz[0], y=elbow_xyz[1], z=elbow_xyz[2],
                 visibility=vis_elb, presence=1.0),
        Landmark(x=wrist_xyz[0], y=wrist_xyz[1], z=wrist_xyz[2],
                 visibility=vis_wrs, presence=1.0),
    ]

def _make_bilateral_frame(l_shoulder, l_elbow, l_wrist, l_vis,
                          r_shoulder, r_elbow, r_wrist, r_vis, frame_num=0):
    landmarks = [Landmark(0,0,0,0,0)] * 33   # dummy base
    # left limb indices (example: 11,13,15)
    landmarks[11] = Landmark(*l_shoulder, visibility=l_vis, presence=1.0)
    landmarks[13] = Landmark(*l_elbow, visibility=l_vis, presence=1.0)
    landmarks[15] = Landmark(*l_wrist, visibility=l_vis, presence=1.0)
    # right limb indices (12,14,16)
    landmarks[12] = Landmark(*r_shoulder, visibility=r_vis, presence=1.0)
    landmarks[14] = Landmark(*r_elbow, visibility=r_vis, presence=1.0)
    landmarks[16] = Landmark(*r_wrist, visibility=r_vis, presence=1.0)
    return FrameData(frame_number=frame_num, timestamp_s=0.0,
                     landmarks=landmarks, world_landmarks=landmarks[:])


def build_frame_from_limbs(left_limb_landmarks, right_limb_landmarks=None):
    """
    Creates a FrameData where world_landmarks are arranged so that:
      - left limb occupies positions that correspond to the limb indices.
      - right limb (if given) occupies the other side indices.
    All other landmarks are dummy zeros (visibility 0.0) to not interfere.
    """
    from landmark_dicts import get_landmark_indices_from_exercise
    # Get the actual limb index tuples for a bilateral bicep curl (or any bilateral)
    left_limb, right_limb_indices = get_landmark_indices_from_exercise("bicep_curl")
    # left_limb = (11,13,15) or similar; right_limb = (12,14,16)
    # Build world_landmarks list of 33 Landmarks with zeros
    world = [Landmark(0,0,0, visibility=0.0, presence=0.0) for _ in range(33)]
    # Place left limb landmarks
    for idx, lm in zip(left_limb, left_limb_landmarks):
        world[idx] = lm
    if right_limb_landmarks is not None:
        for idx, lm in zip(right_limb_indices, right_limb_landmarks):
            world[idx] = lm
    # Return a FrameData with the same landmarks in both landmarks and world_landmarks,
    # because derive_angles only uses world_landmarks via get_limb.
    return FrameData(frame_number=0, timestamp_s=0.0, landmarks=world, world_landmarks=world)


def build_exercise(
    name="bicep_curl",
    bilateral="bilateral",
    limbs=None,
    frame_data=None
):
    """Constructs an exercise object with minimal required fields."""
    return exercise(
        name=name,
        limbs=limbs if limbs is not None else [(0,1,2), (3,4,5)],  # dummy
        bilateral=bilateral,
        is_flexion=True,
        frame_data=frame_data if frame_data is not None else []
    )

def test_calculate_angle_right_angle():
    a = Landmark(-1.0, 0.0, 0.0, visibility=1.0, presence=1.0)
    b = Landmark(0.0, 0.0, 0.0, visibility=1.0, presence=1.0)
    c = Landmark(0.0, 1.0, 0.0, visibility=1.0, presence=1.0)
    assert pytest.approx(_calculate_angle(a, b, c), rel=1e-6) == 90.0


def test_calculate_angle_zero_angle():
    # collinear: a -> b -> c along same axis (positive x)
    a = Landmark(1.0, 0.0, 0.0, 1.0, 1.0)
    b = Landmark(0.0, 0.0, 0.0, 1.0, 1.0)
    c = Landmark(2.0, 0.0, 0.0, 1.0, 1.0)
    assert _calculate_angle(a, b, c) == 0.0


def test_calculate_angle_180_degrees():
    # a->b->c form straight line in opposite directions
    a = Landmark(-1.0, 0.0, 0.0, 1.0, 1.0)
    b = Landmark(0.0, 0.0, 0.0, 1.0, 1.0)
    c = Landmark(1.0, 0.0, 0.0, 1.0, 1.0)
    assert pytest.approx(_calculate_angle(a, b, c), rel=1e-6) == 180.0


def test_calculate_angle_division_by_zero():
    # Same points cause zero-length vectors -> ValueError
    a = Landmark(0.0, 0.0, 0.0, 1.0, 1.0)
    b = Landmark(0.0, 0.0, 0.0, 1.0, 1.0)
    c = Landmark(0.0, 0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="Cannot compute angle: one of the vectors has zero length."):
        _calculate_angle(a, b, c)

def test_derive_angles_unilateral_left_visible():
    """Left limb fully visible -> returns calculated angle."""
    left_limb = make_limb_landmarks(
        shoulder_xyz=(-1,0,0),
        elbow_xyz=(0,0,0),
        wrist_xyz=(0,1,0),
        vis_shld=0.9, vis_elb=0.9, vis_wrs=0.9
    )
    frame = build_frame_from_limbs(left_limb)
    ex = build_exercise(bilateral="left", frame_data=[frame])
    # For unilateral, limbs[0] is the only tuple; need to set it correctly
    from landmark_dicts import get_landmark_indices_from_exercise
    left_limb_idx = get_landmark_indices_from_exercise("bicep_curl")[0]
    ex.limbs = [left_limb_idx]  # unilateral left

    angles = derive_angles(ex)
    assert len(angles) == 1
    assert not np.isnan(angles[0])
    assert pytest.approx(angles[0], rel=1e-5) == 90.0  # right angle at elbow


def test_derive_angles_unilateral_left_not_visible():
    """One joint visibility <=0.7 -> result is NaN."""
    left_limb = make_limb_landmarks(
        shoulder_xyz=(-1,0,0),
        elbow_xyz=(0,0,0),
        wrist_xyz=(0,1,0),
        vis_shld=0.5,  # below threshold
        vis_elb=1.0,
        vis_wrs=1.0
    )
    frame = build_frame_from_limbs(left_limb)
    from landmark_dicts import get_landmark_indices_from_exercise
    ex = build_exercise(bilateral="left", frame_data=[frame])
    ex.limbs = [get_landmark_indices_from_exercise("bicep_curl")[0]]

    angles = derive_angles(ex)
    assert len(angles) == 1
    assert np.isnan(angles[0])

def test_derive_angles_bilateral_both_visible():
    """Both sides visible -> returns weighted average of the two angles."""
    left_limb = make_limb_landmarks(
        shoulder_xyz=(-1,0,0),
        elbow_xyz=(0,0,0),
        wrist_xyz=(0,1,0),
        vis_shld=0.9, vis_elb=0.9, vis_wrs=0.9
    )
    right_limb = make_limb_landmarks(
        shoulder_xyz=(1,0,0),
        elbow_xyz=(0,0,0),
        wrist_xyz=(0,1,0),
        vis_shld=0.8, vis_elb=0.8, vis_wrs=0.8
    )
    frame = build_frame_from_limbs(left_limb, right_limb)
    ex = build_exercise(bilateral="bilateral", frame_data=[frame])
    ex.limbs = get_landmark_indices_from_exercise("bicep_curl")  # two tuples

    angles = derive_angles(ex)
    assert len(angles) == 1
    assert not np.isnan(angles[0])
    # Both limbs produce 90°, so weighted average is also 90°
    assert pytest.approx(angles[0], rel=1e-5) == 90.0


def test_derive_angles_bilateral_only_left_visible():
    """Right side not visible enough -> falls back to left angle."""
    left_limb = make_limb_landmarks(
        shoulder_xyz=(-1,0,0),
        elbow_xyz=(0,0,0),
        wrist_xyz=(0,1,0),
        vis_shld=1.0, vis_elb=1.0, vis_wrs=1.0
    )
    right_limb = make_limb_landmarks(
        shoulder_xyz=(1,0,0),
        elbow_xyz=(0,0,0),
        wrist_xyz=(0,1,0),
        vis_shld=0.5, vis_elb=0.5, vis_wrs=0.5  # below threshold
    )
    frame = build_frame_from_limbs(left_limb, right_limb)
    ex = build_exercise(bilateral="bilateral", frame_data=[frame])
    ex.limbs = get_landmark_indices_from_exercise("bicep_curl")

    angles = derive_angles(ex)
    assert len(angles) == 1
    assert pytest.approx(angles[0], rel=1e-5) == 90.0


def test_derive_angles_bilateral_only_right_visible():
    """Left side not visible enough -> falls back to right angle."""
    left_limb = make_limb_landmarks(
        shoulder_xyz=(-1,0,0),
        elbow_xyz=(0,0,0),
        wrist_xyz=(0,1,0),
        vis_shld=0.5, vis_elb=0.5, vis_wrs=0.5
    )
    right_limb = make_limb_landmarks(
        shoulder_xyz=(1,0,0),
        elbow_xyz=(0,0,0),
        wrist_xyz=(0,1,0),
        vis_shld=1.0, vis_elb=1.0, vis_wrs=1.0
    )
    frame = build_frame_from_limbs(left_limb, right_limb)
    ex = build_exercise(bilateral="bilateral", frame_data=[frame])
    ex.limbs = get_landmark_indices_from_exercise("bicep_curl")

    angles = derive_angles(ex)
    assert len(angles) == 1
    assert pytest.approx(angles[0], rel=1e-5) == 90.0


def test_derive_angles_bilateral_neither_visible():
    """Both sides below threshold -> NaN."""
    left_limb = make_limb_landmarks(
        shoulder_xyz=(-1,0,0),
        elbow_xyz=(0,0,0),
        wrist_xyz=(0,1,0),
        vis_shld=0.5, vis_elb=0.5, vis_wrs=0.5
    )
    right_limb = make_limb_landmarks(
        shoulder_xyz=(1,0,0),
        elbow_xyz=(0,0,0),
        wrist_xyz=(0,1,0),
        vis_shld=0.5, vis_elb=0.5, vis_wrs=0.5
    )
    frame = build_frame_from_limbs(left_limb, right_limb)
    ex = build_exercise(bilateral="bilateral", frame_data=[frame])
    ex.limbs = get_landmark_indices_from_exercise("bicep_curl")

    angles = derive_angles(ex)
    assert len(angles) == 1
    assert np.isnan(angles[0])


def test_derive_angles_multiple_frames():
    """Angles list length matches number of frames."""
    left_limb = make_limb_landmarks(
        shoulder_xyz=(-1,0,0),
        elbow_xyz=(0,0,0),
        wrist_xyz=(0,1,0),
        vis_shld=1.0, vis_elb=1.0, vis_wrs=1.0
    )
    frame1 = build_frame_from_limbs(left_limb)
    frame2 = build_frame_from_limbs(left_limb)
    from landmark_dicts import get_landmark_indices_from_exercise
    ex = build_exercise(bilateral="left", frame_data=[frame1, frame2])
    ex.limbs = [get_landmark_indices_from_exercise("bicep_curl")[0]]

    angles = derive_angles(ex)
    assert len(angles) == 2
    assert all(not np.isnan(a) for a in angles)
    # Both frames identical => both angles equal 90°
    for a in angles:
        assert pytest.approx(a, rel=1e-5) == 90.0

class TestBilateralAngle:
    def test_both_sides_visible_averages(self):
        # Both sides set to 90° angle (using same coordinates)
        # 90° -> wrist at (1,0,0) relative to elbow at origin, shoulder (0,-1,0)
        elbow = (0,0,0)
        shoulder = (0,-1,0)
        wrist = (1,0,0)
        frame = _make_bilateral_frame(
            shoulder, elbow, wrist, 0.9,
            shoulder, elbow, wrist, 0.8,
        )
        exercise = mock_exercise_bilateral([(11,13,15), (12,14,16)], [frame])
        angles = derive_angles(exercise)
        assert len(angles) == 1
        # Both should be 90°, weighted avg = (90*0.9 + 90*0.8)/(1.7) = 90
        assert angles[0] == pytest.approx(90.0)

    def test_one_side_low_visibility_uses_other(self):
        # Left visibility 0.9, right 0.6 (<0.7) → use left angle only
        frame = _make_bilateral_frame(
            (0,-1,0), (0,0,0), (1,0,0), 0.9,   # 90°
            (0,-1,0), (0,0,0), (0,1,0), 0.6,   # 180° (straight arm? but irrelevant)
        )
        exercise = mock_exercise_bilateral([(11,13,15), (12,14,16)], [frame])
        angles = derive_angles(exercise)
        assert len(angles) == 1
        # Only left used -> 90°
        assert angles[0] == pytest.approx(90.0)

    def test_both_low_visibility_gives_nan(self):
        frame = _make_bilateral_frame(
            (0,-1,0), (0,0,0), (1,0,0), 0.5,
            (0,-1,0), (0,0,0), (1,0,0), 0.6,
        )
        exercise = mock_exercise_bilateral([(11,13,15), (12,14,16)], [frame])
        angles = derive_angles(exercise)
        assert len(angles) == 1
        assert np.isnan(angles[0])

def mock_exercise_bilateral(limb_indices, frames):
    """Helper that builds a minimal exercise object for bilateral testing."""
    class Exercise:
        def __init__(self):
            self.frame_data = frames
            self.limbs = [limb_indices[0], limb_indices[1]]
            self.bilateral = "bilateral"
    return Exercise()