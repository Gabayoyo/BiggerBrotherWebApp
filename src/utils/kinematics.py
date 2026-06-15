from dto.frame_data import Landmark
from exercise import exercise
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

def _calculate_velocity(prev: Landmark, curr: Landmark, dt: float) -> np.ndarray:
    if dt <= 0:
        raise ValueError("dt must be positive")

    displacement = curr.to_array() - prev.to_array()
    return displacement / dt

def derive_angles(exercise: exercise) -> list[float]:

    angles = []

    for frame in exercise.frame_data:

        # get one joint you know is visible

        joint1 = frame.get_world_landmark(exercise.limbs[0][0])
        joint2 = frame.get_world_landmark(exercise.limbs[0][1])
        joint3 = frame.get_world_landmark(exercise.limbs[0][2])

        if exercise.bilateral == "bilateral":
            # For bilateral exercises, we can choose one side (e.g., left) for angle calculation
            joint4 = frame.get_world_landmark(exercise.limbs[1][0])
            joint5 = frame.get_world_landmark(exercise.limbs[1][1])
            joint6 = frame.get_world_landmark(exercise.limbs[1][2])

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

def derive_velocity(
    smoothed_positions: np.ndarray,   # shape (n_frames, 3)
    timestamps: np.ndarray,           # shape (n_frames,)
    fill_first: bool = True           # if True, first entry = None; else first = 0 or NaN
) -> np.ndarray:
    """
    Compute 3D velocity from pre‑smoothed position data.
    
    Returns:
        velocities: shape (n_frames, 3) or (n_frames-1, 3) depending on fill_first.
        If fill_first=True, velocities[0] = [nan, nan, nan].
    """
    n = len(smoothed_positions)
    if len(timestamps) != n:
        raise ValueError("positions and timestamps must have same length")
    
    # Central difference for interior points, forward/backward at edges
    velocities = np.full_like(smoothed_positions, np.nan)
    
    # For i=0 (first frame) – leave as NaN or set to zero? NaN is safer.
    # We'll compute using forward difference for i=0 and backward for i=-1 if needed,
    # but typical approach: use np.gradient which handles edges.
    # However, np.gradient on smoothed data is fine:
    dt = np.gradient(timestamps)
    velocities = np.gradient(smoothed_positions, dt, axis=0, edge_order=2)
    
    if fill_first:
        # velocities[0] already nan if timestamps[0] has zero gradient? Actually np.gradient gives a value.
        # Let's force first to NaN if user wants alignment with original (first frame has no velocity).
        velocities[0] = np.nan
    return velocities