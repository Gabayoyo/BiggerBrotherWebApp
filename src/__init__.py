# src/__init__.py
# exposes the main functions of the package for easy import

# Main functions for analysis
from analysis.pose_estimator import PoseEstimator # class for estimating pose data from videos
from analysis.compute_metrics import compute_metrics # main metric pipeline function
from analysis.compute_rep_metrics import compute_rep_metrics # function for computing rep-level metrics from list of angles

# Key DTOs
from dto.frame_data import FrameData, Landmark
from dto.results import RepAnalysisResult, RirAnalysisResult

# Main class for pipeline
from bigger_brother import BiggerBrother

__all__ = [
    "PoseEstimator",
    "compute_metrics",
    "compute_rep_metrics",
    "FrameData",
    "Landmark",
    "RepAnalysisResult",
    "RirAnalysisResult",
    "BiggerBrother"
]