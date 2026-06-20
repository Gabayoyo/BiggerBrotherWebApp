from dataclasses import dataclass

from utils.utils import sanitise_exercise_input, sanitise_unilateral_input

@dataclass
class InputConfig:
    exercise: str
    weight: float
    laterality: str = "bilateral"
    visualise: bool = False
    visualise_curve: bool = False

    def __post_init__(self):
        # Validate / normalise immediately
        self.exercise = sanitise_exercise_input(self.exercise)
        self.laterality = sanitise_unilateral_input(self.laterality)