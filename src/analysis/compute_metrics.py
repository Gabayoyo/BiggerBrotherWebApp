# file that contains all functions computing metrics from the pose estimation output, such as rep-level features
# i.e. rep detection, concentric/eccentric phase segmentation, ROM calculation, speed calculation, etc.

from utils.velocity import derive_velocity
from dto.frame_data import FrameData
from dto.results import RepMetric
from analysis.visualiser import animate_skeleton
from analysis.compute_rep import compute_reps
from utils.angle import derive_angles
from utils.utils import smooth_floats
from dto.exercise import get_exercise
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from utils.velocity import derive_velocity

from utils.velocity import derive_velocity


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
    metrics = compute_reps(smoothed_angles, fps, is_flexion=exercise_info.is_flexion)

    velocities = derive_velocity(exercise_info, fps)

    updated_metrics = []
    for rep in metrics:
        start = rep.concentric_start_frame
        end   = rep.concentric_end_frame
        if start <= end and end < len(velocities):
            peak = max(velocities[start : end + 1])
        else:
            peak = None   # or 0.0
        updated_metrics.append(replace(rep, peak_concentric_speed_ms=peak))

    return updated_metrics