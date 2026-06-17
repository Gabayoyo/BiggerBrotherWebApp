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
    exercise_info = get_exercise(exercise, laterality, frame_data)

    # find ROM, angular velocity and mean velocity -> rep metrics
    angles = derive_angles(exercise_info)

    # angle is smoothed afterwards; is better
    smoothed_angles = smooth_floats(np.array(angles))

    if visualise:
        anim = animate_skeleton(frame_data)

    # count reps, returns rep metrics
    metrics = compute_reps(smoothed_angles, fps, is_flexion=exercise_info.is_flexion)

    # returns velocities of each frame, which is used to compute peak concentric speed for each rep
    velocities = derive_velocity(exercise_info, fps)

    updated_metrics = []
    for rep in metrics:
        if rep.con_start_frame <= rep.con_end_frame and rep.con_end_frame < len(velocities):
            # Extract the concentric velocity slice
            segment = velocities[rep.con_start_frame : rep.con_end_frame + 1]
            # Calculate the mean (average) instead of the max
            mean_vel = sum(segment) / len(segment) if len(segment) > 0 else 0.0
        else:
            mean_vel = 0.0

        updated_metrics.append(replace(rep, mean_concentric_speed_ms=mean_vel))

    return updated_metrics