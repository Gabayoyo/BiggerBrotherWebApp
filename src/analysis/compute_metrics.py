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

# core function that computes metrics from the pose estimation output
def compute_metrics(frame_data: list[FrameData], visualise: bool, exercise: str, laterality: str, fps: float) -> list[RepMetric]:

    # construct exercise instance from exercise name and laterality, which contains the relevant landmarks for angle derivation
    exercise_info = get_exercise(exercise, laterality, frame_data)

    # find ROM, angular velocity and mean velocity -> rep metrics
    angles = derive_angles(exercise_info)

    # angle is smoothed afterwards - is better for angles
    smoothed_angles = smooth_floats(np.array(angles))

    if visualise:
        anim = animate_skeleton(frame_data)

    # count reps, returns rep metrics
    metrics = compute_reps(smoothed_angles, fps, is_flexion=exercise_info.is_flexion)

    # returns velocities of each frame
    velocities = derive_velocity(exercise_info, fps)

    # get mean concentric speed for each rep and update the metrics
    updated_metrics = []
    for rep in metrics:

        if rep.con_start_frame <= rep.con_end_frame and rep.con_end_frame < len(velocities):
            segment = velocities[rep.con_start_frame : rep.con_end_frame + 1]
            mean_vel = sum(segment) / len(segment) if len(segment) > 0 else 0.0
        else:
            mean_vel = 0.0

        updated_metrics.append(replace(rep, mean_concentric_speed_ms=mean_vel))

    return updated_metrics