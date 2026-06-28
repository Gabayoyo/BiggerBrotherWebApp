from dataclasses import dataclass

import numpy as np

from landmark_dicts import LANDMARK_INDICES


# a single landmark point in 3D space, with visibility and presence information
@dataclass
class Landmark:
    x: float
    y: float
    z: float  # depth; estimated
    visibility: float  # how likely it's visible
    presence: float  # if landmark is present in frame

    # checks if landmark is confidently visible
    @property
    def visible(self) -> bool:
        return self.visibility > 0.7

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_mediapipe(cls, landmark):
        return cls(
            x=landmark.x,
            y=landmark.y,
            z=landmark.z,
            visibility=landmark.visibility,
            presence=landmark.presence,
        )


# data class representing pose data for a single video frame
@dataclass
class FrameData:
    frame_number: int
    timestamp_s: float  # seconds from start
    landmarks: list[Landmark]  # 2D landmarks (normalized)
    world_landmarks: list[Landmark]  # 3D landmarks (meters, relative to world origin)

    def __post_init__(self):
        if len(self.landmarks) != 33:
            raise ValueError(f"Expected 33 landmarks, got {len(self.landmarks)}")

    # get landmark by index or name
    def get_landmark(self, index: int) -> Landmark:
        return self.landmarks[index]

    # get landmark by readable name
    def get_landmark_by_name(self, name: str) -> Landmark:
        index: int = LANDMARK_INDICES[name]  # explicit int type for mypy
        return self.landmarks[index]

    # get world landmark by index or name
    def get_world_landmark(self, index: int) -> Landmark:
        return self.world_landmarks[index]

    # get world landmark by readable name
    def get_world_landmark_by_name(self, name: str) -> Landmark:
        index: int = LANDMARK_INDICES[name]  # explicit int type for mypy
        return self.world_landmarks[index]

    # get all landmarks as a numpy array (33, 3)
    def get_keypoints_array(self) -> np.ndarray:
        return np.array([lm.to_array() for lm in self.landmarks])

    # get all landmarks with visibility > 0.7
    def get_visible_landmarks(self) -> list[tuple[int, Landmark]]:
        return [(i, lm) for i, lm in enumerate(self.landmarks) if lm.visibility > 0.7]

    # get the 3 landmarks given a tuple of indices representing a limb (e.g., shoulder, elbow, wrist)
    def get_limb(
        self, joints: tuple[int, int, int]
    ) -> tuple[Landmark, Landmark, Landmark]:
        return (
            self.get_world_landmark(joints[0]),
            self.get_world_landmark(joints[1]),
            self.get_world_landmark(joints[2]),
        )

    # get the xyz position and visibility of a landmark by name
    def get_lm_xyzv_by_name(self, name: str) -> tuple[np.ndarray, float]:
        lm = self.get_world_landmark_by_name(name)
        return lm.to_array(), lm.visibility

    @classmethod
    def from_mediapipe(
        cls, frame_num: int, timestamp: float, pose_landmarks, world_landmarks
    ):
        landmarks = [Landmark.from_mediapipe(lm) for lm in pose_landmarks.landmark]
        return cls(
            frame_number=frame_num,
            timestamp_s=timestamp,
            landmarks=landmarks,
            world_landmarks=world_landmarks,
        )
