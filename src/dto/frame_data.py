from dataclasses import dataclass
from dto.landmark_dict import LANDMARK_INDICES
import numpy as np

@dataclass
class Landmark:
    """A single 3D landmark point"""
    x: float      # Normalized x coordinate (0-1)
    y: float      # Normalized y coordinate (0-1)
    z: float      # Depth (relative to torso center)
    visibility: float  # 0-1, how likely it's visible
    presence: float    # 0-1, if landmark is present in frame

    @property
    def visible(self) -> bool:
        """Whether landmark is confidently visible."""
        return self.visibility > 0.7
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_mediapipe(cls, landmark):
        """Convert MediaPipe landmark to our Landmark class"""
        return cls(
            x=landmark.x,
            y=landmark.y,
            z=landmark.z,
            visibility=landmark.visibility,
            presence=landmark.presence
        )


@dataclass
class FrameData:
    """Pose data for a single video frame"""
    frame_number: int
    timestamp_s: float  # Seconds from start
    landmarks: list[Landmark]  # 2D landmarks (normalized)
    world_landmarks: list[Landmark] # 3D landmarks (meters, relative to world origin)
    
    def __post_init__(self):
        """Validate landmarks count"""
        if len(self.landmarks) != 33:
            raise ValueError(f"Expected 33 landmarks, got {len(self.landmarks)}")
    
    def get_landmark(self, index: int) -> Landmark:
        """Get landmark by MediaPipe index (0-32)"""
        return self.landmarks[index]
    
    def get_landmark_by_name(self, name: str) -> Landmark:
        """Get landmark by readable name"""
        return self.landmarks[LANDMARK_INDICES[name]]
    
    def get_keypoints_array(self) -> np.ndarray:
        """Get all landmarks as numpy array (33, 3)"""
        return np.array([lm.to_array() for lm in self.landmarks])
    
    def get_visible_landmarks(self) -> list[tuple[int, Landmark]]:
        """Return only landmarks with high visibility"""
        return [(i, lm) for i, lm in enumerate(self.landmarks) if lm.visibility > 0.5]
    
    @classmethod
    def from_mediapipe(cls, frame_num: int, timestamp: float, pose_landmarks):
        """Create FrameData from MediaPipe pose_landmarks object"""
        landmarks = [
            Landmark.from_mediapipe(lm) 
            for lm in pose_landmarks.landmark
        ]
        return cls(
            frame_number=frame_num,
            timestamp=timestamp,
            landmarks=landmarks
        )