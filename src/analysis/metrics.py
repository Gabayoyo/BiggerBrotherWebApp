# file that contains all functions computing metrics from the pose estimation output, such as rep-level features
# i.e. rep detection, concentric/eccentric phase segmentation, ROM calculation, speed calculation, etc.

from dto.frame_data import FrameData
from dto.results import RepMetrics


def compute_metrics(frame_data: list[FrameData]) -> RepMetrics:
    """
    Compute metrics from the pose estimation output.
    """
    # count reps -> rep boundaries
    # find ROM, angular velocity and mean velocity -> rep metrics
    print(frame_data)
    pass