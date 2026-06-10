import argparse

from src.dto.results import RepAnalysisResult, RepMetrics, RirAnalysisResult
from pathlib import Path
from model import ensure_model

CACHE_DIR = Path("./cache")

class BiggerBrother:

    def __init__(self, args: argparse.Namespace):
        self.input_path = Path(args.input_path) if args.input_path else None
        self.calibration_path = Path(args.calibration_path) if args.calibration_path else None
        self.cache_dir = Path(args.cache_dir) if args.cache_dir else CACHE_DIR
        self.model_path = Path(args.model_path) if args.model_path else ensure_model()
        self.pose_estimator = PoseEs
        self.cache_outputs = args.cache_outputs if args.cache_outputs else False

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

    # def list_processed_videos(self) -> list[Path]:

    def run(self):
        print(self.input_path)
        print(self.calibration_path)
        print(self.model_path)
        print(self.cache_dir)
        print(self.cache_outputs)

def main():
    parser = argparse.ArgumentParser(
        description="Pose-based form analysis and RiR estimation"
    )

    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input video"
    )

    parser.add_argument(
        "--calibration_path",
        type=str,
        default=None,
        help="Path to calibration video"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to pose landmarker model. If not provided, will use default and download if necessary."
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to store cached files. Defaults to './cache'."
    )

    parser.add_argument(
        "--cache_outputs", 
        action="store_true"
    )

    args = parser.parse_args()

    app = BiggerBrother(args)
    app.run()
    
if __name__ == "__main__":
    main()