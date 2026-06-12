from dto.frame_data import FrameData, Landmark
from dto.landmark_dict import ANGLES
import numpy as np

def calculate_angle(a: Landmark, b: Landmark, c: Landmark) -> float:
    """
    Calculate the angle at point b, formed by points a-b-c, in 3D.
    Returns angle in degrees.
    """
    ba = a.to_array() - b.to_array()
    bc = c.to_array() - b.to_array()

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # guard against floating point drift

    return np.degrees(np.arccos(cosine_angle))

def derive_angles(frame_data: list[FrameData]) -> list[float]:

    angles = []

    for frame in frame_data:
        if (shoulder.visible and elbow.visible and wrist.visible):
            shoulder = frame.get_landmark_by_name("LEFT_SHOULDER")
            elbow = frame.get_landmark_by_name("LEFT_ELBOW")
            wrist = frame.get_landmark_by_name("LEFT_WRIST")
            angle = calculate_angle(shoulder, elbow, wrist)
            angles.append(angle)
        else:
            angles.append(None)  # Append None if any landmark is not above visibility threshold
    return angles