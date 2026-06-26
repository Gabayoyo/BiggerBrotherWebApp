import sys
import os
from pathlib import Path
import numpy as np
import cv2
import pytest

from analysis.pose_estimator import PoseEstimator
from model import ensure_model

pytestmark = pytest.mark.integration


def create_test_video(path, num_frames=10, fps=30, size=(100, 100)):
    """Create a synthetic .mp4 file with num_frames of blue frames."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
    )

    frame_data, fps = estimator.process_video(video_path, frame_skip=1)

    # Basic sanity checks – a solid blue video may not produce any poses
    assert fps == 30.0
    assert isinstance(frame_data, list)
    for fd in frame_data:
        assert hasattr(fd, 'landmarks')
        assert len(fd.landmarks) == 33