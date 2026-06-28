import numpy as np

from dto.exercise import Exercise
from dto.frame_data import Landmark


# helper function that calculates the angle between three landmarks in 3D space
def _calculate_angle(a: Landmark, b: Landmark, c: Landmark) -> float:
    ba = a.to_array() - b.to_array()
    bc = c.to_array() - b.to_array()

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # check for zero-length vectors to avoid division by zero
    if norm_ba == 0.0 or norm_bc == 0.0:
        raise ValueError("Cannot compute angle: one of the vectors has zero length.")

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))


# main function to derive angles from exercise/frame data
def derive_angles(exercise: Exercise) -> list[float]:

    angles = []

    # iterate through frames, get the relevant landmarks for the exercise, and compute angles
    for frame in exercise.frame_data:
        # get one joint you know is visible
        # first entry is always limb of interest whether left or right, second entry is the other side if bilateral
        joint1, joint2, joint3 = frame.get_limb(exercise.limbs[0])

        if exercise.bilateral == "bilateral":
            # for bilateral exercises, we get second index which is the other limb
            joint4, joint5, joint6 = frame.get_limb(exercise.limbs[1])

            # get angles and confidence
            conf_left = min(joint1.visibility, joint2.visibility, joint3.visibility)
            conf_right = min(joint4.visibility, joint5.visibility, joint6.visibility)
            angle_left = _calculate_angle(joint1, joint2, joint3)
            angle_right = _calculate_angle(joint4, joint5, joint6)

            # if both sides are visible, we can average the angles weighted by confidence
            if conf_left > 0.7 and conf_right > 0.7:
                weighted_angle = (angle_left * conf_left + angle_right * conf_right) / (
                    conf_left + conf_right
                )
                angles.append(weighted_angle)
            elif conf_left > 0.7:
                angles.append(
                    angle_left
                )  # append left angle if right side is not sufficiently visible
            elif conf_right > 0.7:
                angles.append(angle_right)  # and vice versa
            else:
                angles.append(
                    np.nan
                )  # append NaN if neither side is sufficiently visible

        elif joint1.visible and joint2.visible and joint3.visible:
            # else if unilateral, we can use the specified side's landmarks and check visibility
            # which again, will always be the first entry in the limbs list
            angle = _calculate_angle(joint1, joint2, joint3)
            angles.append(angle)
        else:
            angles.append(
                np.nan
            )  # append NaN if any of the two angles is not above visibility threshold
    return angles
