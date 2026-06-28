from unittest import mock

import numpy as np
import pytest

from bigger_brother import BiggerBrother
from dto.frame_data import FrameData, Landmark
from dto.results import RepAnalysisResult

pytestmark = pytest.mark.e2e

NUM_LANDMARKS = 33


def _landmark(x=0.0, y=0.0, z=0.0, visibility=1.0, presence=1.0):
    return Landmark(x=x, y=y, z=z, visibility=visibility, presence=presence)


def _dummy_landmarks(override_dict=None):
    landmarks = [_landmark(x=i / 100, y=i / 100, z=0) for i in range(NUM_LANDMARKS)]
    if override_dict:
        for idx, lm in override_dict.items():
            landmarks[idx] = lm
    return landmarks


def _frame_with_angle(angle_deg, frame_num=1, timestamp=0.0):
    """Right-arm elbow angle: shoulder=12, elbow=14, wrist=16."""
    rad = np.radians(angle_deg)
    wrist_x = np.sin(rad)
    wrist_y = -np.cos(rad)

    elbow = (0.0, 0.0, 0.0)
    shoulder = (0.0, -1.0, 0.0)
    wrist = (wrist_x, wrist_y, 0.0)

    overrides = {
        12: _landmark(*shoulder),
        14: _landmark(*elbow),
        16: _landmark(*wrist),
    }
    landmarks = _dummy_landmarks(overrides)
    world_landmarks = landmarks[:]

    return FrameData(
        frame_number=frame_num,
        timestamp_s=timestamp,
        landmarks=landmarks,
        world_landmarks=world_landmarks,
    )


def _build_rep_frames():
    """20 frames forming one clear bicep curl rep: 30° → 150° → 30°."""
    flat = 5
    ramp = 5
    angles = (
        [30] * flat
        + list(np.linspace(30, 150, ramp))
        + list(np.linspace(150, 30, ramp))
        + [30] * flat
    )
    return [_frame_with_angle(a, i, i / 10.0) for i, a in enumerate(angles)], 10.0


@pytest.fixture
def video_path(tmp_path):
    """Dummy video path (file doesn't need to exist)."""
    return tmp_path / "test_video.mp4"


@pytest.fixture
def bb_instance(tmp_path):
    """BiggerBrother with a temporary cache directory."""
    cache = tmp_path / "cache"
    return BiggerBrother(cache_dir=cache, cache_data=False)


class TestAnalyseReps:
    def test_happy_path(self, bb_instance, video_path, input_config):
        """Full pipeline returns a RepAnalysisResult with detected rep."""
        frames, fps = _build_rep_frames()

        with mock.patch.object(
            bb_instance.pose_estimator, "process_video", return_value=(frames, fps)
        ):
            result = bb_instance.analyse_reps(video_path, input_config)

        assert isinstance(result, RepAnalysisResult)
        assert result.video_path == video_path
        assert result.exercise == "bicep_curl"
        assert len(result.metrics) == 1
        rep = result.metrics[0]
        assert rep.rep_number == 1
        assert rep.mean_concentric_speed_ms is not None
        assert rep.mean_concentric_speed_ms > 0

    def test_too_few_frames_raises(self, bb_instance, video_path, input_config):
        """Passing fewer than MIN_FRAMES frames should raise ValueError."""
        frames = [_frame_with_angle(80, i, i * 0.1) for i in range(10)]  # 10 frames
        with (
            mock.patch.object(
                bb_instance.pose_estimator, "process_video", return_value=(frames, 10.0)
            ),
            pytest.raises(ValueError, match="At least 20 frames"),
        ):
            bb_instance.analyse_reps(video_path, input_config)

    def test_visualise_flag_does_not_crash(
        self, bb_instance, video_path, input_config, monkeypatch
    ):
        """Visualise=True suppresses plot window and runs without error."""
        monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
        input_config.visualise = True
        frames, fps = _build_rep_frames()

        with mock.patch.object(
            bb_instance.pose_estimator, "process_video", return_value=(frames, fps)
        ):
            result = bb_instance.analyse_reps(video_path, input_config)

        assert isinstance(result, RepAnalysisResult)
        assert len(result.metrics) == 1

    def test_invalid_exercise_raises(self, bb_instance, video_path, input_config):
        """An unknown exercise name propagates the error from get_exercise."""
        input_config.exercise = "nonexistent_exercise"
        frames, fps = _build_rep_frames()

        with (
            mock.patch.object(
                bb_instance.pose_estimator, "process_video", return_value=(frames, fps)
            ),
            pytest.raises(ValueError),
        ):
            bb_instance.analyse_reps(video_path, input_config)
