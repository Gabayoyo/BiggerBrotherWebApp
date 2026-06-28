import math

import numpy as np

from dto.exercise import Exercise
from dto.frame_data import FrameData, Landmark
from landmark_dicts import LANDMARK_INDICES, LANDMARK_OF_INTEREST
from utils.velocity import derive_velocity

np.random.seed(0)


def make_velocity_frames(
    exercise_name: str,
    bilateral: str,
    positions_left: list[tuple[float, float, float]],
    positions_right: list[tuple[float, float, float]] = None,
    vis_left: list[float] = None,
    vis_right: list[float] = None,
    fps: float = 30.0,
):
    """
    Creates a list of FrameData objects for velocity testing.

    Args:
        exercise_name: e.g., "bicep_curl"
        bilateral: "left", "right", "bilateral"
        positions_left: list of (x,y,z) for the left wrist per frame.
        positions_right: required if bilateral or right; else None.
        vis_left: list of visibility values per frame for left wrist (default 1.0).
        vis_right: list of visibility values per frame for right wrist (default 1.0).
        fps: frames per second (used for timestamp spacing).
    """
    left_name, right_name = LANDMARK_OF_INTEREST[exercise_name]
    left_idx = LANDMARK_INDICES[left_name]
    right_idx = LANDMARK_INDICES[right_name]

    n_frames = len(positions_left)
    if vis_left is None:
        vis_left = [1.0] * n_frames
    if vis_right is None:
        vis_right = [1.0] * n_frames

    frames = []
    for i in range(n_frames):
        # Build 33 landmarks, all zero visibility initially
        world = [Landmark(0, 0, 0, visibility=0.0, presence=0.0) for _ in range(33)]
        # Left wrist
        xl, yl, zl = positions_left[i]
        world[left_idx] = Landmark(xl, yl, zl, visibility=vis_left[i], presence=1.0)
        # Right wrist if needed
        if bilateral in ("bilateral", "right"):
            xr, yr, zr = positions_right[i]
            world[right_idx] = Landmark(
                xr, yr, zr, visibility=vis_right[i], presence=1.0
            )
        frames.append(
            FrameData(
                frame_number=i,
                timestamp_s=i / fps,
                landmarks=world,
                world_landmarks=world,
            )
        )
    return frames


def make_exercise(name, bilateral, frames, is_flexion=True):
    """Helper to construct an exercise with the correct limb indices."""
    from landmark_dicts import get_landmark_indices_from_exercise

    limbs = get_landmark_indices_from_exercise(name)
    if bilateral != "bilateral":
        side = 0 if bilateral == "left" else 1
        limbs = [limbs[side]]
    return Exercise(
        name=name,
        limbs=limbs,
        bilateral=bilateral,
        is_flexion=is_flexion,
        frame_data=frames,
    )


def test_derive_velocity_too_few_frames():
    """<2 frames → returns list of zeros."""
    frames = make_velocity_frames("bicep_curl", "left", [(1, 2, 3)], fps=30.0)
    ex = make_exercise("bicep_curl", "left", frames)
    vel = derive_velocity(ex, fps=30.0)
    assert vel == [0.0]


def test_derive_velocity_unilateral_left_high_confidence():
    """Left wrist moves linearly at constant speed → velocity ≈ speed."""
    # 0.1 m per frame at 30 fps → speed = 3.0 m/s
    n = 25
    pos = [(0.1 * i, 0.0, 0.0) for i in range(n)]
    frames = make_velocity_frames(
        "bicep_curl", "left", pos, vis_left=[1.0] * n, fps=30.0
    )
    ex = make_exercise("bicep_curl", "left", frames)
    vel = derive_velocity(ex, fps=30.0)
    # Interior values (excluding boundaries) should be around 3.0 m/s
    interior = vel[5:-5]  # avoid filter edge effects
    assert 2.5 <= np.mean(interior) <= 3.5
    assert all(v >= 0 for v in vel)


def test_derive_velocity_unilateral_left_low_confidence():
    """Low visibility → NaN pos → interpolated → velocity computed."""
    n = 25
    pos = [(0.1 * i, 0.0, 0.0) for i in range(n)]
    vis = [1.0] * n
    vis[10] = 0.3  # below threshold at frame 10
    frames = make_velocity_frames("bicep_curl", "left", pos, vis_left=vis, fps=30.0)
    ex = make_exercise("bicep_curl", "left", frames)
    vel = derive_velocity(ex, fps=30.0)
    assert not any(math.isnan(v) for v in vel)
    interior = vel[5:-5] if len(vel) >= 10 else vel
    # Should still be close to 3.0 m/s
    assert 2.5 < np.mean(interior) < 3.5


def test_derive_velocity_unilateral_right_high_confidence():
    """Right wrist moving → velocity computed."""
    n = 25
    # 0.2 m per frame in y, 30 fps → speed = 6.0 m/s
    pos_right = [(0.0, 0.2 * i, 0.0) for i in range(n)]
    pos_left = [(0, 0, 0)] * n  # not used
    frames = make_velocity_frames(
        "bicep_curl", "right", pos_left, pos_right, vis_right=[1.0] * n, fps=30.0
    )
    ex = make_exercise("bicep_curl", "right", frames)
    vel = derive_velocity(ex, fps=30.0)
    interior = vel[5:-5] if len(vel) >= 10 else vel
    assert 5.0 <= np.mean(interior) <= 7.0


def test_derive_velocity_bilateral_both_high_confidence_equal_weight():
    """Both wrists move identically → weighted average = same motion → speed."""
    n = 25
    pos = [(0.1 * i, 0.0, 0.0) for i in range(n)]  # 3.0 m/s
    frames = make_velocity_frames(
        "bicep_curl",
        "bilateral",
        pos,
        pos,
        vis_left=[1.0] * n,
        vis_right=[1.0] * n,
        fps=30.0,
    )
    ex = make_exercise("bicep_curl", "bilateral", frames)
    vel = derive_velocity(ex, fps=30.0)
    interior = vel[5:-5] if len(vel) >= 10 else vel
    assert 2.5 <= np.mean(interior) <= 3.5


def test_derive_velocity_bilateral_one_low_confidence():
    """One side has low visibility → weighted, velocity still near left side's speed."""
    n = 25
    pos_left = [(0.1 * i, 0, 0) for i in range(n)]  # 3.0 m/s
    pos_right = [(0, 0, 0) for i in range(n)]  # stationary
    vis_left = [1.0] * n
    vis_right = [0.2] * n  # low confidence
    frames = make_velocity_frames(
        "bicep_curl",
        "bilateral",
        pos_left,
        pos_right,
        vis_left=vis_left,
        vis_right=vis_right,
        fps=30.0,
    )
    ex = make_exercise("bicep_curl", "bilateral", frames)
    vel = derive_velocity(ex, fps=30.0)
    interior = vel[5:-5] if len(vel) >= 10 else vel
    # Weighted average: heavily toward left, so speed ~3 m/s
    assert 2.0 < np.mean(interior) < 4.0


def test_derive_velocity_bilateral_zero_confidence():
    """
    Both sides zero confidence → positions become NaN → interpolated/zeroed → velocity zero.
    """
    n = 25
    pos = [(0.1 * i, 0, 0) for i in range(n)]
    frames = make_velocity_frames(
        "bicep_curl",
        "bilateral",
        pos,
        pos,
        vis_left=[0.0] * n,
        vis_right=[0.0] * n,
        fps=30.0,
    )
    ex = make_exercise("bicep_curl", "bilateral", frames)
    vel = derive_velocity(ex, fps=30.0)
    assert all(abs(v) < 1e-6 for v in vel)


def test_derive_velocity_all_frames_invisible_unilateral():
    """All frames low confidence → positions zeroed → velocity zero."""
    n = 25
    pos = [(0.1 * i, 0.2 * i, 0.3 * i) for i in range(n)]
    frames = make_velocity_frames(
        "bicep_curl", "left", pos, vis_left=[0.0] * n, fps=30.0
    )
    ex = make_exercise("bicep_curl", "left", frames)
    vel = derive_velocity(ex, fps=30.0)
    # After zeroing out, positions constant → velocity zero (or near zero due to filtering noise)
    assert all(abs(v) < 1e-3 for v in vel)


def test_derive_velocity_constant_position():
    """No motion → velocity zero (or negligible)."""
    n = 25
    pos = [(1, 2, 3)] * n
    frames = make_velocity_frames("bicep_curl", "left", pos, fps=30.0)
    ex = make_exercise("bicep_curl", "left", frames)
    vel = derive_velocity(ex, fps=30.0)
    # Filtering may cause tiny fluctuations, but essentially zero
    assert all(abs(v) < 1e-3 for v in vel)


def test_derive_velocity_linear_motion_known_speed():
    """Uniform linear motion → velocity magnitude constant."""
    n = 25
    # 0.1 m/frame * 30 fps = 3.0 m/s
    pos = [(0.1 * i, 0.0, 0.0) for i in range(n)]
    frames = make_velocity_frames("bicep_curl", "left", pos, fps=30.0)
    ex = make_exercise("bicep_curl", "left", frames)
    vel = derive_velocity(ex, fps=30.0)
    interior = vel[5:-5] if len(vel) >= 10 else vel
    assert 2.5 <= np.mean(interior) <= 3.5


def test_derive_velocity_boundary_frames():
    """Boundary frames are finite and positive for forward motion."""
    n = 25
    pos = [(0.1 * i, 0.0, 0.0) for i in range(n)]
    frames = make_velocity_frames("bicep_curl", "left", pos, fps=30.0)
    ex = make_exercise("bicep_curl", "left", frames)
    vel = derive_velocity(ex, fps=30.0)
    assert not np.isnan(vel[0])
    assert not np.isnan(vel[-1])
    assert vel[0] > 0
    assert vel[-1] > 0
