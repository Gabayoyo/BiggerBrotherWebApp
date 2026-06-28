import argparse

from analysis.compute_VL_curve import compute_VL_curve
from dto.input_config import InputConfig
from dto.results import RepAnalysisResult, RirAnalysisResult
from analysis.pose_estimator import PoseEstimator
from pathlib import Path
from model import ensure_model
from analysis.compute_metrics import compute_metrics
from analysis.estimate_rir import estimate_rir_from_curve

CACHE_DIR = Path("./cache")

# main class for the BiggerBrother pipeline, which handles pose estimation, rep analysis, and RiR estimation
# def run() purely for CLI usage
# doubles as a service class with exposed methods for streamlit or library imports
class BiggerBrother:

    def __init__(self, cache_dir: Path = None, cache_data: bool = False):
        self.model_path = ensure_model()
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_data = cache_data
        self.pose_estimator = PoseEstimator(
            self.model_path,
            cache_dir=self.cache_dir,
            cache_data=self.cache_data
        )

    # ANALYSE_REPS "ENDPOINT"
    # given frame data from pose estimation, returns a RepAnalysisResult with rep metrics
    def analyse_reps(self, video_path: Path, input_config: InputConfig) -> RepAnalysisResult:
        frame_data, fps = self.pose_estimator.process_video(video_path)
        
        metrics = compute_metrics(
            frame_data, 
            visualise=input_config.visualise, 
            exercise=input_config.exercise, 
            laterality=input_config.laterality, 
            fps=fps
        )

        return RepAnalysisResult(video_path=video_path, exercise=input_config.exercise, metrics=metrics)

    # ESTIMATE_RIR "ENDPOINT"
    def estimate_rir(self, target_video_path: Path, calibration_video_path: Path, input_config: InputConfig) -> RirAnalysisResult:
        # get load-velocity profile
        # plot kg to velocity
        # this finds 1RM
        # VL (%) = 100 × (V_BEST - V_LAST) / V_BEST
        # converts this velocity loss to percentage of repetitions performed
        # RIR = (Reps Completed / % Reps Performed) - Reps Completed

        frame_data, fps = self.pose_estimator.process_video(calibration_video_path)
        
        calibration_metrics = compute_metrics(
            frame_data, 
            visualise=input_config.visualise, 
            exercise=input_config.exercise, 
            laterality=input_config.laterality, 
            fps=fps
        )
        
        coeffs = compute_VL_curve(calibration_metrics, visualise_curve=input_config.visualise_curve)
        estimated_rir = estimate_rir_from_curve(calibration_metrics, coeffs)
        return RirAnalysisResult(video_path=target_video_path, metrics=calibration_metrics, rir_estimate=estimated_rir)

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

    parser.add_argument(
        "--visualise-curve",
        action="store_true",
        help="Visualise the load-velocity curve"
    )

    args = parser.parse_args()

    service = BiggerBrother(
        cache_dir=None,
        cache_data=args.cache_data
    )

    config = InputConfig(
        exercise=args.exercise,
        weight=args.weight,
        laterality=args.laterality,
        visualise=args.visualise,
        visualise_curve=args.visualise_curve
    )

    # if calibration, estimate rir...

    rir = service.estimate_rir(
        target_video_path=Path(args.input_path),
        calibration_video_path=Path(args.input_path),
        input_config=config
    )

    if args.input_path:
        rep_analysis_result = service.analyse_reps(Path(args.input_path), config)
        print(rep_analysis_result.console_output())
    
if __name__ == "__main__":
    main()