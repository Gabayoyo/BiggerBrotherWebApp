# file that contains all functions computing metrics from the pose estimation output, such as rep-level features
# i.e. rep detection, concentric/eccentric phase segmentation, ROM calculation, speed calculation, etc.

from dto.frame_data import FrameData
from dto.results import RepMetrics
from analysis.visualiser import animate_skeleton
from utils.kinematics import derive_angles
from utils.utils import smooth_floats, smooth_landmark_trajectory
from dto.landmark_dict import LANDMARK_INDICES

import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(frame_data: list[FrameData], visualise: bool) -> RepMetrics:
    """
    Compute metrics from the pose estimation output.
    """
    # find ROM, angular velocity and mean velocity -> rep metrics
    angles = derive_angles(frame_data)
    # angle is smoothed afterwards; is better
    smoothed_angles = smooth_floats(np.array(angles))

    frames = range(len(smoothed_angles)) # frame indices

    # count reps, return rep boundaries

    # landmarks are smoothed before velocity calculation to reduce noise
    smoothed_landmarks = smooth_landmark_trajectory(frame_data, landmark_index=LANDMARK_INDICES['LEFT_WRIST'])
    pass