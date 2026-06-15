# file that contains all functions computing metrics from the pose estimation output, such as rep-level features
# i.e. rep detection, concentric/eccentric phase segmentation, ROM calculation, speed calculation, etc.

from dto.frame_data import FrameData
from dto.results import RepMetrics
from analysis.visualiser import animate_skeleton
from utils.kinematics import derive_angles
from utils.utils import smooth_floats, smooth_landmark_trajectory
from landmark_dicts import LANDMARK_INDICES
from exercises import get_exercise

import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(frame_data: list[FrameData], visualise: bool, exercise: str, unilateral: str) -> RepMetrics:
    """
    Compute metrics from the pose estimation output.
    """
    # get more visible side
    exercise_info = get_exercise(exercise, unilateral, frame_data)

    # find ROM, angular velocity and mean velocity -> rep metrics
    angles = derive_angles(exercise_info)

    for angle in angles:
        print(angle)
    # angle is smoothed afterwards; is better
    smoothed_angles = smooth_floats(np.array(angles))

    if visualise:
        anim = animate_skeleton(frame_data)

    # count reps, return rep boundaries

    # landmarks are smoothed before velocity calculation to reduce noise
    smoothed_landmarks = smooth_landmark_trajectory(frame_data, landmark_index=LANDMARK_INDICES['LEFT_WRIST'])
    pass