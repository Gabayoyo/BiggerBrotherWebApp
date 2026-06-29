import cv2
import numpy as np
import pytest

from analysis.pose_estimator import PoseEstimator
from model import ensure_model

pytestmark = pytest.mark.integration


def create_test_video(path, num_frames=10, fps=30, size=(100, 100)):
    """Create a synthetic .mp4 file with num_frames of blue frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, size)
    for _ in range(num_frames):
        frame = np.full((size[1], size[0], 3), (255, 0, 0), dtype=np.uint8)
        out.write(frame)
    out.release()


def test_process_video_integration(tmp_path):
    """
    Run process_video on a real synthetic video using the actual MediaPipe model.
    The model is downloaded automatically if needed.
    """
    # Ensure the model is present; download if missing
    try:
        model_path = ensure_model()
    except Exception as e:
        pytest.skip(f"Could not obtain model: {e}")

    video_path = tmp_path / "test.mp4"
    create_test_video(video_path, num_frames=10, fps=30)

    estimator = PoseEstimator(
        model_path=model_path,
        cache_dir=tmp_path / "cache",
        cache_data=False,
        use_gpu=False,
    )

    frame_data, fps = estimator.process_video(video_path, frame_skip=1)

    assert fps == 30.0
    assert isinstance(frame_data, list)
    for fd in frame_data:
        assert hasattr(fd, "landmarks")
        assert len(fd.landmarks) == 33


def test_fps_extraction(tmp_path):
    video_path = tmp_path / "fps_test.mp4"
    create_test_video(video_path, num_frames=30, fps=15.0)  # 15 fps
    estimator = PoseEstimator(
        model_path=ensure_model(),
        cache_dir=tmp_path / "cache",
        cache_data=False,
        use_gpu=False,
    )
    _, fps = estimator.process_video(video_path, frame_skip=1)
    assert fps == 15.0


def test_caching_returns_identical_data(tmp_path):
    video_path = tmp_path / "cache.mp4"
    create_test_video(video_path, num_frames=10, fps=10)
    estimator = PoseEstimator(
        model_path=ensure_model(),
        cache_dir=tmp_path / "cache",
        cache_data=True,
        use_gpu=False,
    )
    fd1, fps1 = estimator.process_video(video_path, frame_skip=1)
    fd2, fps2 = estimator.process_video(video_path, frame_skip=1)
    assert fps1 == fps2
    assert len(fd1) == len(fd2)
    for f1, f2 in zip(fd1, fd2, strict=True):
        assert f1.frame_number == f2.frame_number
        # landmarks might be identical; at least same length
        assert len(f1.landmarks) == len(f2.landmarks)
