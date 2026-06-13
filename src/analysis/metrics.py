# file that contains all functions computing metrics from the pose estimation output, such as rep-level features
# i.e. rep detection, concentric/eccentric phase segmentation, ROM calculation, speed calculation, etc.

from dto.frame_data import FrameData
from dto.results import RepMetrics
from utils.kinematics import derive_angles
from utils.utils import smooth_floats, smooth_landmark_trajectory
from dto.landmark_dict import LANDMARK_INDICES

import numpy as np
import pandas as pd


def compute_metrics(frame_data: list[FrameData]) -> RepMetrics:
    """
    Compute metrics from the pose estimation output.
    """
    # count reps -> rep boundaries
    # find ROM, angular velocity and mean velocity -> rep metrics
    angles = derive_angles(frame_data)
    smoothed_angles = smooth_floats(np.array(angles))
    smoothed_landmarks = smooth_landmark_trajectory(frame_data, landmark_index=LANDMARK_INDICES['LEFT_WRIST'])
    pass