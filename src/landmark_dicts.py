# MediaPipe landmark indices (0-32)
LANDMARK_INDICES = {
    # face
    "NOSE": 0,
    "LEFT_EYE_INNER": 1,
    "LEFT_EYE": 2,
    "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4,
    "RIGHT_EYE": 5,
    "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7,
    "RIGHT_EAR": 8,
    "LEFT_MOUTH": 9,
    "RIGHT_MOUTH": 10,
    # torso
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17,
    "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19,
    "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21,
    "RIGHT_THUMB": 22,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    # legs
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29,
    "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
}

# triplets of landmarks for angle calculations (joint angles)
ANGLE = {
    "ELBOW_LEFT": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "ELBOW_RIGHT": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    "SHOULDER_LEFT": ("LEFT_HIP", "LEFT_SHOULDER", "LEFT_ELBOW"),
    "SHOULDER_RIGHT": ("RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"),
    "HIP_LEFT": ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
    "HIP_RIGHT": ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
}

# which angles we are interested in for a given exercise when computing angle/ROM
ANGLES_OF_INTEREST = {
    "tricep_extension": ["ELBOW_LEFT", "ELBOW_RIGHT"],
    "bicep_curl": ["ELBOW_LEFT", "ELBOW_RIGHT"],
}

# which points we are interested in for a given exercise when computing velocity
LANDMARK_OF_INTEREST = {
    "tricep_extension": ["LEFT_WRIST", "RIGHT_WRIST"],
    "bicep_curl": ["LEFT_WRIST", "RIGHT_WRIST"],
}


# given an angle name (e.g., 'ELBOW_LEFT'), return the corresponding landmark indices.
def get_landmark_indices_from_angle(angle_name: str) -> tuple[int, int, int]:

    if angle_name not in ANGLE:
        raise ValueError(f"Angle '{angle_name}' not defined in ANGLE dictionary.")

    landmark_names = ANGLE[angle_name]
    return tuple(LANDMARK_INDICES[name] for name in landmark_names)


# given an exercise name, return the corresponding landmark indices for all angles of interest.
def get_landmark_indices_from_exercise(
    exercise_name: str,
) -> list[tuple[int, int, int]]:

    if exercise_name not in ANGLES_OF_INTEREST:
        raise ValueError(
            f"Exercise '{exercise_name}' not defined in ANGLES_OF_INTEREST dictionary."
        )

    angle_names = ANGLES_OF_INTEREST[exercise_name]
    return [get_landmark_indices_from_angle(angle_name) for angle_name in angle_names]
