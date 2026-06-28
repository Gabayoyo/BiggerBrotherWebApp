import pytest
import numpy as np
from dto.frame_data import FrameData, Landmark
from analysis.compute_metrics import compute_metrics

pytestmark = pytest.mark.integration

NUM_LANDMARKS = 33

def _landmark(x=0.0, y=0.0, z=0.0, visibility=1.0, presence=1.0):
    return Landmark(x=x, y=y, z=z, visibility=visibility, presence=presence)

def _dummy_landmarks(override_dict=None):
    landmarks = [_landmark(x=i/100, y=i/100, z=0) for i in range(NUM_LANDMARKS)]
    if override_dict:
        for idx, lm in override_dict.items():
            landmarks[idx] = lm
    return landmarks

def _frame_with_angle(angle_deg, frame_num=1, timestamp=0.0,
                      shoulder_idx=12, elbow_idx=14, wrist_idx=16):   # right-arm indices
    """
    Create a FrameData whose elbow angle is exactly `angle_deg`.
    Elbow at origin, shoulder at (0,-1), wrist placed to achieve the angle.
    Uses MediaPipe right‑arm indices by default.
    """
    rad = np.radians(angle_deg)
    wrist_x = np.sin(rad)
    wrist_y = -np.cos(rad)          # cos(angle) = -y

    elbow = (0.0, 0.0, 0.0)
    shoulder = (0.0, -1.0, 0.0)
    wrist = (wrist_x, wrist_y, 0.0)

    overrides = {
        shoulder_idx: _landmark(*shoulder),
        elbow_idx:    _landmark(*elbow),
        wrist_idx:    _landmark(*wrist),
    }
    landmarks = _dummy_landmarks(overrides)
    world_landmarks = landmarks[:]      # copy for world coordinates

    return FrameData(
        frame_number=frame_num,
        timestamp_s=timestamp,
        landmarks=landmarks,
        world_landmarks=world_landmarks,
    )

def _make_frames(angles, fps=10.0):
    """Turn a list of angles into a list of FrameData with timestamps."""
    return [_frame_with_angle(a, i, i/fps) for i, a in enumerate(angles)]

class TestComputeMetricsIntegration:
    def test_single_flexion_rep(self):
        """One full rep: low → high → low, 20 frames with smooth variation."""
        # Build a continuous-looking waveform: 5 flat 30°, ramp up to 150°, ramp down to 30°, 5 flat
        flat = 5
        ramp = 5
        angles = (
            [30] * flat +
            list(np.linspace(30, 150, ramp)) +
            list(np.linspace(150, 30, ramp)) +
            [30] * flat
        )  # total 20 frames
        frames = _make_frames(angles, fps=10.0)
        metrics = compute_metrics(frames, visualise=False, exercise="bicep_curl",
                                  laterality="right", fps=10.0)
        assert len(metrics) == 1, f"Expected 1 rep, got {len(metrics)}"
        rep = metrics[0]
        assert rep.rep_number == 1
        assert rep.con_start_frame <= rep.con_end_frame
        assert rep.ecc_start_frame <= rep.ecc_end_frame
        assert rep.mean_concentric_speed_ms > 0

    def test_two_reps_flexion(self):
        """Two full reps, 20 frames."""
        # Two up-down cycles with a little flat region at start/end
        angles = (
            [30] * 3 +
            list(np.linspace(30, 150, 3)) +
            list(np.linspace(150, 30, 3)) +
            list(np.linspace(30, 150, 3)) +
            list(np.linspace(150, 30, 3)) +
            [30] * 3
        )  # total 18 frames -> pad to 20 with extra 30's
        angles += [30] * (20 - len(angles))
        frames = _make_frames(angles, fps=10.0)
        metrics = compute_metrics(frames, False, "bicep_curl", "right", 10.0)
        assert len(metrics) >= 2, f"Expected at least 2 reps, got {len(metrics)}"

    def test_visualise_flag_does_not_crash(self, monkeypatch):
        """Visualise=True runs without opening a plot window, 20 frames."""
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda: None)
        angles = [30] * 20  # flat, enough for filters
        frames = _make_frames(angles, fps=5.0)
        metrics = compute_metrics(frames, visualise=True, exercise="bicep_curl",
                                  laterality="right", fps=5.0)
        assert isinstance(metrics, list)

    def test_minimal_input_returns_no_reps(self):
        """20 frames with constant angle → 0 reps."""
        angles = [80] * 20
        frames = _make_frames(angles, fps=10.0)
        metrics = compute_metrics(frames, False, "bicep_curl", "right", 10.0)
        assert metrics == []

    def test_velocity_segment_clipping(self):
        """
        Last rep ends with a peak and slight descent but no trough.
        The fallback sets concentric end to the last frame.
        """
        # Create a signal: start low, rise to peak, then drop a bit and hold.
        # Enough frames to avoid filter crashes.
        angles = (
            [50] * 6 +
            [80, 110, 140, 130, 120, 120, 120] +
            [120] * (20 - 13)
        )
        frames = _make_frames(angles, fps=10.0)
        metrics = compute_metrics(frames, False, "bicep_curl", "right", 10.0)
        if len(metrics) > 0:
            rep = metrics[0]
            # con_end_frame should be the last frame (index 19)
            assert rep.con_end_frame == 19
            assert rep.mean_concentric_speed_ms is not None
            assert rep.mean_concentric_speed_ms >= 0.0

    def test_small_bump_not_detected_as_rep(self):
        """A tiny fluctuation (prominence < 30) should yield zero reps."""
        # Create a signal that goes 80 → 85 → 80 over 20 frames – tiny wiggle
        angles = [80] * 10 + [82, 85, 82] + [80] * 7
        angles = angles[:20]  # ensure exactly 20
        frames = _make_frames(angles, fps=10.0)
        metrics = compute_metrics(frames, False, "bicep_curl", "right", 10.0)
        assert metrics == []