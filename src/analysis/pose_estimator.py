import hashlib
import pickle
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from dto.frame_data import FrameData, Landmark


class PoseEstimator:
    def __init__(
        self,
        model_path: Path,
        cache_dir: Path = Path(".cache"),
        cache_data: bool = False,
        use_gpu: bool = False,
    ):
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.cache_data = cache_data
        self.use_gpu = use_gpu

    def _cache_key(self, video_path: Path, frame_skip: int) -> str:
        stat = video_path.stat()
        fingerprint = f"{video_path}{stat.st_size}{stat.st_mtime}{frame_skip}"
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def _make_options(self):
        delegate = (
            mp_python.BaseOptions.Delegate.GPU
            if self.use_gpu
            else mp_python.BaseOptions.Delegate.CPU
        )
        return mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(self.model_path),
                delegate=delegate,
            ),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            output_segmentation_masks=False,
        )

    def process_video(
        self, video_path: Path, frame_skip: int = 2
    ) -> tuple[list[FrameData], float]:

        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

        cache_path = self.cache_dir / f"{self._cache_key(video_path, frame_skip)}.pkl"
        if cache_path.exists():
            return pickle.loads(cache_path.read_bytes()), fps

        options = self._make_options()
        frame_data_list: list[FrameData] = []
        frame_idx = 0

        try:
            with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
                while True:
                    success, frame = capture.read()  # single combined call
                    if not success:
                        break

                    if frame_idx % frame_skip != 0:
                        frame_idx += 1
                        continue

                    rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGBA, data=rgba_frame
                    )
                    timestamp_ms = int(frame_idx * 1000 / fps)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                    if result.pose_landmarks:
                        frame_data_list.append(
                            FrameData(
                                frame_number=frame_idx,
                                timestamp_s=timestamp_ms / 1000,
                                landmarks=[
                                    Landmark.from_mediapipe(lm)
                                    for lm in result.pose_landmarks[0]
                                ],
                                world_landmarks=[
                                    Landmark.from_mediapipe(lm)
                                    for lm in result.pose_world_landmarks[0]
                                ],
                            )
                        )

                    frame_idx += 1
        finally:
            capture.release()

        if self.cache_data:
            cache_path.parent.mkdir(exist_ok=True)
            cache_path.write_bytes(pickle.dumps(frame_data_list))

        return frame_data_list, fps
