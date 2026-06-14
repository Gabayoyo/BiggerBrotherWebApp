from landmark_dicts import ANGLE, get_landmark_indices_from_exercise
from dto.frame_data import FrameData

class exercise:
    def __init__(self, name: str, limbs: list[tuple[int, int, int]], bilateral: str, is_flexion: bool, frame_data: list[FrameData] = None):
        self.name = name
        self.limbs = limbs
        self.bilateral = bilateral
        self.is_flexion = is_flexion # True if flexion, False if extension
        self.frame_data = frame_data # List of frame data for the exercise
        
_EXERCISE_CONFIGS = {
    "tricep_extension": {
        "uni_possible": True, # True if the exercise can be performed unilaterally
        "is_flexion": False,
    },
    "bicep_curl": {
        "uni_possible": True,
        "is_flexion": True,
    },
}

def get_exercise(name: str, bilateral: str) -> exercise:
    name = name.strip().lower()

    if name not in _EXERCISE_CONFIGS:
        raise ValueError(f"Unknown exercise: {name}")
    
    config = _EXERCISE_CONFIGS[name]

    if bilateral != "bilateral" and not config["uni_possible"]:
        raise ValueError(f"Exercise '{name}' cannot be performed unilaterally.")
    
    limbs = get_landmark_indices_from_exercise(name)

    # Select the limb(s) based on the bilateral setting
    if bilateral != "bilateral":
        side_index = 0 if bilateral == "left" else 1
        limbs = [limbs[side_index]]

    return exercise(
        name=name,
        limbs=limbs,
        bilateral=bilateral,
        is_flexion=config["is_flexion"],
    )