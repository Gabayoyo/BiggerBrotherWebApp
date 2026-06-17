import argparse

from dto.frame_data import FrameData
from dto.results import RepAnalysisResult, RirAnalysisResult
from analysis.pose_estimator import PoseEstimator
from pathlib import Path
from model import ensure_model
from analysis.compute_metrics import compute_metrics
from utils.utils import sanitise_exercise_input, sanitise_unilateral_input

CACHE_DIR = Path("./cache")

# main class for the BiggerBrother pipeline, which handles pose estimation, rep analysis, and RiR estimation
# def run() purely for CLI usage
# doubles as a service class with exposed methods for streamlit or library imports
class BiggerBrother:

    def __init__(self, args: argparse.Namespace):
        self.input_path = Path(args.input_path) if args.input_path else None
        self.weight = args.weight
        self.exercise = sanitise_exercise_input(args.exercise)
        self.laterality = sanitise_unilateral_input(args.laterality)
        self.calibration_path = Path(args.calibration_path) if args.calibration_path else None
        self.cache_dir = Path(args.cache_dir) if args.cache_dir else CACHE_DIR
        self.model_path = ensure_model()
        self.cache_data = args.cache_data if args.cache_data else False
        self.visualise = args.visualise if args.visualise else False
        self.pose_estimator = PoseEstimator(self.model_path, cache_dir=self.cache_dir, cache_data=self.cache_data)

    # ANALYSE_REPS "ENDPOINT"
    # given frame data from pose estimation, returns a RepAnalysisResult with rep metrics
    def analyse_reps(self, video_path: Path) -> RepAnalysisResult:
        frame_data, fps = self.pose_estimator.process_video(video_path)
        metrics = compute_metrics(frame_data, visualise=self.visualise, exercise=self.exercise, laterality=self.laterality, fps=fps)
        return RepAnalysisResult(video_path=self.input_path, exercise=self.exercise, metrics=metrics)

    # ESTIMATE_RIR "ENDPOINT"
    def estimate_rir(
        self,
        target_video: Path,
        failure_video: Path,
    ) -> RirAnalysisResult:
        pass

    # run() method for CLI usage
    def run(self):
        # if self.calibration_path:
        #   calibration_result = self.pose_estimator.process_video(self.calibration_path)
        #   velocity_estimate = self.calculate_velocity(calibration_result)

        # process target video and analyze reps
        if self.input_path:
            rep_analysis_result = self.analyse_reps(self.input_path)
            print(rep_analysis_result.console_output())

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
        "exercise",
        type=str,
        help="Type of exercise being performed in the video (e.g. 'bicep curl', 'squat', etc.)"
    )

    parser.add_argument(
        "weight",
        type=float,
        help="Weight used in the exercise (in kg)"
    )

    parser.add_argument(
        "--laterality",
        type=str,
        default="bilateral",
        help="Specify if the exercise is unilateral (e.g. 'left' or 'right')"
    )

    parser.add_argument(
        "--calibration-path",
        type=str,
        default=None,
        help="Path to calibration video"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to store cached files. Defaults to './cache'."
    )

    parser.add_argument(
        "--cache-data", 
        action="store_true"
    )

    parser.add_argument(
        "--visualise",
        action="store_true",
        help="Visualise the pose estimation results"
    )

    args = parser.parse_args()

    app = BiggerBrother(args)
    app.run()
    
if __name__ == "__main__":
    main()