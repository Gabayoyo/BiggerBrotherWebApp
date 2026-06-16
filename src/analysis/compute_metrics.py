# file that contains all functions computing metrics from the pose estimation output, such as rep-level features
# i.e. rep detection, concentric/eccentric phase segmentation, ROM calculation, speed calculation, etc.

from dto.frame_data import FrameData
from dto.results import RepMetric
from analysis.visualiser import animate_skeleton
from analysis.compute_rep_metrics import compute_rep_metrics
from utils.kinematics import derive_angles
from utils.utils import smooth_floats
from dto.exercise import get_exercise

import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(frame_data: list[FrameData], visualise: bool, exercise: str, laterality: str, fps: float) -> list[RepMetric]:
    """
    Compute metrics from the pose estimation output.
    """
    # get more visible side
    exercise_info = get_exercise(exercise, laterality, frame_data)

    # find ROM, angular velocity and mean velocity -> rep metrics
    angles = derive_angles(exercise_info)

    # angle is smoothed afterwards; is better
    smoothed_angles = smooth_floats(np.array(angles))

    if visualise:
        anim = animate_skeleton(frame_data)

    # count reps, return rep metrics
    metrics = compute_rep_metrics(smoothed_angles, fps, is_flexion=exercise_info.is_flexion)

    # metrics = compute_velocity_metrics(metrics, frame_data, exercise_info, fps)

    return metrics