import argparse
from pathlib import Path

import numpy as np

from analysis.compute_1rm import compute_1rm
from analysis.compute_metrics import compute_metrics
from analysis.compute_vl_curve import compute_vl_curve as _compute_vl_curve
from analysis.estimate_rir import estimate_rir_from_curve
from analysis.pose_estimator import PoseEstimator
from dto.input_config import InputConfig
from dto.rep_metric import RepMetric
from dto.results import RepAnalysisResult, RirAnalysisResult
from model import ensure_model

CACHE_DIR = Path("./cache")


# main class for the BiggerBrother pipeline, which handles pose estimation, rep analysis, and RiR estimation
# def run() purely for CLI usage
# doubles as a service class with exposed methods for streamlit or library imports
class BiggerBrother:
    def __init__(self, cache_dir: Path | None = None, cache_data: bool = False):
        self.model_path = ensure_model()
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_data = cache_data
        self.pose_estimator = PoseEstimator(
            self.model_path, cache_dir=self.cache_dir, cache_data=self.cache_data
        )

    def get_metrics(
        self, video_path: Path, input_config: InputConfig
    ) -> RepAnalysisResult:
        """Process a video and return per-rep metrics + estimated 1RM.

        Used for both regular analysis and calibration uploads.
        """
        frame_data, fps = self.pose_estimator.process_video(video_path)

        metrics = compute_metrics(
            frame_data,
            visualise=input_config.visualise,
            exercise=input_config.exercise,
            laterality=input_config.laterality,
            fps=fps,
        )

        estimated_1rm = (
            compute_1rm(input_config.weight, len(metrics)) if metrics else 0.0
        )

        return RepAnalysisResult(
            video_path=video_path,
            exercise=input_config.exercise,
            metrics=metrics,
            estimated_1rm=estimated_1rm,
        )

    def compute_vl_curve(
        self,
        calibration_metrics: list[RepMetric],
        visualise_curve: bool = False,
    ) -> np.poly1d:
        """Pure computation: calibration metrics → VL curve coefficients.

        No video I/O — caller passes already-computed metrics.
        """
        return _compute_vl_curve(calibration_metrics, visualise_curve=visualise_curve)  # type: ignore[no-any-return]

    def estimate_rir(
        self,
        target_metrics: list[RepMetric],
        vl_model: np.poly1d,
    ) -> int:
        """Pure computation: target metrics + VL model → reps in reserve.

        No video I/O — caller passes already-computed metrics and model.
        """
        return int(estimate_rir_from_curve(target_metrics, vl_model))


def main():
    parser = argparse.ArgumentParser(
        description="Pose-based form analysis and RiR estimation"
    )

    parser.add_argument("input_path", type=str, help="Path to input video")

    parser.add_argument(
        "exercise",
        type=str,
        help="Type of exercise being performed in the video (e.g. 'bicep curl', 'squat', etc.)",
    )

    parser.add_argument(
        "weight", type=float, help="Weight used in the exercise (in kg)"
    )

    parser.add_argument(
        "--laterality",
        type=str,
        default="bilateral",
        help="Specify if the exercise is unilateral (e.g. 'left' or 'right')",
    )

    parser.add_argument(
        "--calibration-path", type=str, default=None, help="Path to calibration video"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to store cached files. Defaults to './cache'.",
    )

    parser.add_argument("--cache-data", action="store_true")

    parser.add_argument(
        "--visualise", action="store_true", help="Visualise the pose estimation results"
    )

    parser.add_argument(
        "--visualise-curve",
        action="store_true",
        help="Visualise the load-velocity curve",
    )

    args = parser.parse_args()

    service = BiggerBrother(cache_dir=None, cache_data=args.cache_data)

    config = InputConfig(
        exercise=args.exercise,
        weight=args.weight,
        laterality=args.laterality,
        visualise=args.visualise,
        visualise_curve=args.visualise_curve,
    )

    if args.calibration_path:
        # Two-step: calibration → VL curve → target metrics → RiR
        calib_result = service.get_metrics(Path(args.calibration_path), config)
        vl_model = service.compute_vl_curve(calib_result.metrics)
        target_result = service.get_metrics(Path(args.input_path), config)
        rir = service.estimate_rir(target_result.metrics, vl_model)
        rir_result = RirAnalysisResult(
            video_path=Path(args.input_path),
            metrics=target_result.metrics,
            rir_estimate=rir,
            estimated_1rm=target_result.estimated_1rm,
        )
        print(rir_result.summary_table())
    elif args.input_path:
        result = service.get_metrics(Path(args.input_path), config)
        print(result.console_output())


if __name__ == "__main__":
    main()
