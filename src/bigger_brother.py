from dto import RepAnalysisResult, RirAnalysisResult
from pathlib import Path
from analysis.pose_estimator import PoseEstimator

CACHE_DIR = Path("./cache")

class BiggerBrother:

    def __init__(self, cache_dir: Path = CACHE_DIR, pose_model: str = "mediapipe"):
        self.cache_dir = Path(cache_dir)
        self.pose_estimator = PoseEstimator(model_name=pose_model)
        # registry/cache are file-based, so no heavy state, but the paths are configured here.

    # analyses a given video and returns a RepAnalysisResult with rep metrics
    def analyse_form(
        self,
        input_video: str | Path,
    ) -> RepAnalysisResult:
        pass

    def estimate_rir(
        self,
        target_video: str | Path,
        failure_video: str | Path,
    ) -> RirAnalysisResult:
        pass

    def upload_video(
        self,
        video_file
    ) -> Path:
        """Handle video file upload and return the saved file path."""
        pass

    # def rir_from_metrics(
    # self,
    # target_metrics: list[RepMetrics],
    # failure_path: str | Path
    # ) -> RirAnalysisResult:

    # def _process_video(video_path: Path) -> list[RepMetrics]:

    # def list_processed_videos(self) -> list[Path]: