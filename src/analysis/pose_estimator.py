# class responsible for estimating the pose given a CSV using a specified model

from pathlib import Path

from src.dto.results import RepMetrics

class PoseEstimator:
    def __init__(self, model_path: Path, cache_outputs: bool = False):
        self.model_path = model_path
        self.cache_outputs = cache_outputs

    def process_video(self, video_path: Path) -> list[RepMetrics]:
        """
        Process the video and return a list of RepMetrics.
        """
        pass

    def _estimate_frame(self, video_path: Path) -> list[RepMetrics]:
        """
        Estimate the pose from the given video and return a list of dictionaries containing the pose data for each frame.
        """
        pass