from dto.frame_data import FrameData
from landmark_dicts import get_landmark_indices_from_exercise


# main data class representing an exercise and its associated metadata
class Exercise:
    def __init__(
        self,
        name: str,
        limbs: list[tuple[int, int, int]],
        bilateral: str,
        is_flexion: bool,
        frame_data: list[FrameData] = None,
    ):
        self.name = name

        # list of tuples representing the 3 landmark indices for the limb(s) of interest
        # (e.g., shoulder, elbow, wrist = left arm)
        # for unilateral exercises: single tuple that is the side of interest
        # for bilateral exercises, it is a list of two tuples, one for each side
        self.limbs = limbs

        self.bilateral = bilateral  # can be "left", "right", or "bilateral"
        self.is_flexion = is_flexion  # true if flexion, False if extension
        self.frame_data = frame_data  # list of frame data for the exercise


# constructs instance of exercise from name, bilateral setting, and frame data
def get_exercise(name: str, bilateral: str, frame_data: list[FrameData]) -> Exercise:

    if name not in _EXERCISE_CONFIGS:
        raise ValueError(f"Unknown exercise: {name}")

    # get the exercise config for the given name
    config = _EXERCISE_CONFIGS[name]

    # if the exercise is not possible unilaterally, and the user requested a unilateral version, raise an error
    if bilateral != "bilateral" and not config["uni_possible"]:
        raise ValueError(f"Exercise '{name}' cannot be performed unilaterally.")

    # get the landmark indices for the exercise's angles of interest
    limbs = get_landmark_indices_from_exercise(name)

    # select the limb(s) based on the bilateral setting
    if bilateral != "bilateral":
        side_index = 0 if bilateral == "left" else 1
        limbs = [limbs[side_index]]

    return Exercise(
        name=name,
        limbs=limbs,
        bilateral=bilateral,
        is_flexion=config["is_flexion"],
        frame_data=frame_data,
    )


# data map for exercise configs
_EXERCISE_CONFIGS = {
    "tricep_extension": {
        "uni_possible": True,  # True if the exercise can be performed unilaterally
        "is_flexion": False,  # True if the exercise is a flexion movement, False if extension
    },
    "bicep_curl": {
        "uni_possible": True,
        "is_flexion": True,
    },
}
