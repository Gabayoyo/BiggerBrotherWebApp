from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from analysis.pose_estimator import PoseEstimator


def create_mock_video_capture(total_frames, opened=True):
    mock_cap = mock.MagicMock()
    mock_cap.isOpened.return_value = opened
    grab_responses = [True] * total_frames + [False]
    retrieve_responses = [
        (True, np.zeros((10, 10, 3), dtype=np.uint8)) for _ in range(total_frames)
    ]

    def grab_side_effect():
        return grab_responses.pop(0)

    mock_cap.grab.side_effect = grab_side_effect
    mock_cap.retrieve.side_effect = retrieve_responses
    mock_cap.get.return_value = 30.0
    return mock_cap


def create_mock_pose_landmarker(detections_per_frame):
    mock_landmarks = []
    for i in range(33):
        lm = mock.MagicMock()
        lm.x = i / 33.0
        lm.y = i / 33.0
        lm.z = (i % 10) / 100.0
        lm.visibility = 1.0
        lm.presence = 1.0
        mock_landmarks.append(lm)

    def detect_side_effect(mp_image, timestamp_ms):
        if detections_per_frame.pop(0):
            result = mock.MagicMock()
            result.pose_landmarks = [mock_landmarks]
            result.pose_world_landmarks = [mock_landmarks]
            return result
        else:
            result = mock.MagicMock()
            result.pose_landmarks = []
            result.pose_world_landmarks = []
            return result

    mock_landmarker = mock.MagicMock()
    mock_landmarker.detect_for_video.side_effect = detect_side_effect
    return mock_landmarker


# ---------- Fixtures ----------
@pytest.fixture
def pose_estimator(tmp_path):
    model_path = Path("/fake/model.tflite")
    return PoseEstimator(
        model_path=model_path,
        cache_dir=tmp_path / "cache",
        cache_data=True,
    )


@pytest.fixture
def dummy_video_path(tmp_path):
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b"dummy video content")
    return video_file


# ---------- Tests ----------
def test_process_video_happy_path(pose_estimator, dummy_video_path):
    total_frames = 10
    mock_cap = create_mock_video_capture(total_frames)
    mock_landmarker = create_mock_pose_landmarker([True] * total_frames)

    with (
        mock.patch("cv2.VideoCapture", return_value=mock_cap),
        mock.patch("cv2.cvtColor") as mock_cvt,
    ):
        mock_cvt.return_value = np.zeros((10, 10, 4), dtype=np.uint8)
        with mock.patch(
            "analysis.pose_estimator.mp_vision.PoseLandmarker.create_from_options"
        ) as mock_create:
            mock_create.return_value.__enter__.return_value = mock_landmarker
            mock_create.return_value.__exit__ = mock.Mock()

            frame_data, fps = pose_estimator.process_video(
                dummy_video_path, frame_skip=1
            )

    assert fps == 30.0
    assert len(frame_data) == total_frames
    for fd in frame_data:
        assert len(fd.landmarks) == 33


def test_process_video_frame_skip(pose_estimator, dummy_video_path):
    total_frames = 9
    mock_cap = create_mock_video_capture(total_frames)
    mock_landmarker = create_mock_pose_landmarker([True] * (total_frames // 3 + 1))

    with (
        mock.patch("cv2.VideoCapture", return_value=mock_cap),
        mock.patch("cv2.cvtColor", return_value=np.zeros((10, 10, 4), dtype=np.uint8)),
        mock.patch(
            "analysis.pose_estimator.mp_vision.PoseLandmarker.create_from_options"
        ) as mock_create,
    ):
        mock_create.return_value.__enter__.return_value = mock_landmarker
        mock_create.return_value.__exit__ = mock.Mock()

        frame_data, fps = pose_estimator.process_video(dummy_video_path, frame_skip=3)

    assert len(frame_data) == 3
    assert frame_data[0].frame_number == 0
    assert frame_data[1].frame_number == 3
    assert frame_data[2].frame_number == 6


def test_process_video_invalid_frame_skip(pose_estimator, dummy_video_path):
    with pytest.raises(ValueError, match="frame_skip must be >= 1"):
        pose_estimator.process_video(dummy_video_path, frame_skip=0)


def test_process_video_cannot_open(pose_estimator, tmp_path):
    bad_path = tmp_path / "not_exist.mp4"
    mock_cap = mock.MagicMock()
    mock_cap.isOpened.return_value = False
    with (
        mock.patch("cv2.VideoCapture", return_value=mock_cap),
        pytest.raises(ValueError, match="Could not open video"),
    ):
        pose_estimator.process_video(bad_path)


def test_process_video_caching(pose_estimator, dummy_video_path):
    total_frames = 5
    mock_cap = create_mock_video_capture(total_frames)
    mock_landmarker = create_mock_pose_landmarker([True] * total_frames)

    # First call: process and cache
    with (
        mock.patch("cv2.VideoCapture", return_value=mock_cap),
        mock.patch("cv2.cvtColor", return_value=np.zeros((10, 10, 4), dtype=np.uint8)),
        mock.patch(
            "analysis.pose_estimator.mp_vision.PoseLandmarker.create_from_options"
        ) as mock_create,
    ):
        mock_create.return_value.__enter__.return_value = mock_landmarker
        mock_create.return_value.__exit__ = mock.Mock()
        frame_data1, fps1 = pose_estimator.process_video(dummy_video_path)

    # Cache file must now exist
    cache_key = pose_estimator._cache_key(dummy_video_path, 2)
    cache_path = pose_estimator.cache_dir / f"{cache_key}.pkl"
    assert cache_path.exists()

    # Second call: should hit the cache – no frame grabbing/retrieving or MediaPipe use
    # The VideoCapture constructor will be called (to get FPS), but nothing more.
    second_cap = create_mock_video_capture(0)  # won't be used for frames
    second_cap.get.return_value = 30.0  # ensure consistent FPS

    with (
        mock.patch("cv2.VideoCapture", return_value=second_cap) as mock_vc_ctor,
        mock.patch("cv2.cvtColor") as mock_cvt,
        mock.patch(
            "analysis.pose_estimator.mp_vision.PoseLandmarker.create_from_options"
        ) as mock_create,
    ):
        frame_data2, fps2 = pose_estimator.process_video(dummy_video_path)

    # VideoCapture was created to read FPS – allowed
    mock_vc_ctor.assert_called_once()
    # But no frames were grabbed or retrieved
    second_cap.grab.assert_not_called()
    second_cap.retrieve.assert_not_called()
    # No colour conversion or MediaPipe processing
    mock_cvt.assert_not_called()
    mock_create.assert_not_called()

    # Results should match
    assert len(frame_data2) == len(frame_data1)
    assert fps2 == fps1


def test_process_video_no_pose_detected(pose_estimator, dummy_video_path):
    total_frames = 5
    detections = [False, False, True, True, True]
    mock_cap = create_mock_video_capture(total_frames)
    mock_landmarker = create_mock_pose_landmarker(detections)

    with (
        mock.patch("cv2.VideoCapture", return_value=mock_cap),
        mock.patch("cv2.cvtColor", return_value=np.zeros((10, 10, 4), dtype=np.uint8)),
        mock.patch(
            "analysis.pose_estimator.mp_vision.PoseLandmarker.create_from_options"
        ) as mock_create,
    ):
        mock_create.return_value.__enter__.return_value = mock_landmarker
        mock_create.return_value.__exit__ = mock.Mock()

        frame_data, fps = pose_estimator.process_video(dummy_video_path, frame_skip=1)

    assert len(frame_data) == 3
    assert frame_data[0].frame_number == 2
    assert frame_data[1].frame_number == 3
    assert frame_data[2].frame_number == 4
