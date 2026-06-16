from dto.frame_data import Landmark
from dto.exercise import exercise
import numpy as np

def _calculate_angle(a: Landmark, b: Landmark, c: Landmark) -> float:
    """
    Calculate the angle at point b, formed by points a-b-c, in 3D.
    Returns angle in degrees.
    """
    ba = a.to_array() - b.to_array()
    bc = c.to_array() - b.to_array()

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # guard against floating point drift

    return np.degrees(np.arccos(cosine_angle))

def derive_angles(exercise: exercise) -> list[float]:

    angles = []

    for frame in exercise.frame_data:

        # get one joint you know is visible

        joint1, joint2, joint3 = frame.get_limb(exercise.limbs[0])  # e.g., left side

        if exercise.bilateral == "bilateral":
            # For bilateral exercises, we can choose one side (e.g., left) for angle calculation
            joint4, joint5, joint6 = frame.get_limb(exercise.limbs[1])  # right side

            conf_left = min(joint1.visibility, joint2.visibility, joint3.visibility)
            conf_right = min(joint4.visibility, joint5.visibility, joint6.visibility)
            angle_left = _calculate_angle(joint1, joint2, joint3)
            angle_right = _calculate_angle(joint4, joint5, joint6)

            if conf_left > 0.7 and conf_right > 0.7:
                weighted_angle = (angle_left * conf_left + angle_right * conf_right) / (conf_left + conf_right)
                angles.append(weighted_angle)
            elif conf_left > 0.7:
                angles.append(angle_left)
            elif conf_right > 0.7:
                angles.append(angle_right)
            else:
                angles.append(np.nan)  # Append NaN if neither side is sufficiently visible

        elif (joint1.visible and joint2.visible and joint3.visible):
            # else if unilateral, we can use the specified side's landmarks and check visibility
            angle = _calculate_angle(joint1, joint2, joint3)
            angles.append(angle)
        else:
            angles.append(np.nan)  # Append NaN if any of the two angles is not above visibility threshold
    return angles