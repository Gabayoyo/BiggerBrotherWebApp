from dataclasses import dataclass
import numpy as np

@dataclass
class Landmark:
    """A single 3D landmark point"""
    x: float      # Normalized x coordinate (0-1)
    y: float      # Normalized y coordinate (0-1)
    z: float      # Depth (relative to torso center)
    visibility: float  # 0-1, how likely it's visible
    presence: float    # 0-1, if landmark is present in frame
    
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
    landmarks: list[Landmark]  # Length 33
    
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


# MediaPipe landmark indices (0-32)
LANDMARK_INDICES = {
    # Face
    'NOSE': 0,
    'LEFT_EYE_INNER': 1,
    'LEFT_EYE': 2,
    'LEFT_EYE_OUTER': 3,
    'RIGHT_EYE_INNER': 4,
    'RIGHT_EYE': 5,
    'RIGHT_EYE_OUTER': 6,
    'LEFT_EAR': 7,
    'RIGHT_EAR': 8,
    'LEFT_MOUTH': 9,
    'RIGHT_MOUTH': 10,
    
    # Torso
    'LEFT_SHOULDER': 11,
    'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13,
    'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15,
    'RIGHT_WRIST': 16,
    'LEFT_PINKY': 17,
    'RIGHT_PINKY': 18,
    'LEFT_INDEX': 19,
    'RIGHT_INDEX': 20,
    'LEFT_THUMB': 21,
    'RIGHT_THUMB': 22,
    'LEFT_HIP': 23,
    'RIGHT_HIP': 24,
    
    # Legs
    'LEFT_KNEE': 25,
    'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27,
    'RIGHT_ANKLE': 28,
    'LEFT_HEEL': 29,
    'RIGHT_HEEL': 30,
    'LEFT_FOOT_INDEX': 31,
    'RIGHT_FOOT_INDEX': 32,
}

# Common subsets for easier access
UPPER_BODY = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 
              'LEFT_WRIST', 'RIGHT_WRIST']

LOWER_BODY = ['LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 
              'LEFT_ANKLE', 'RIGHT_ANKLE']

SPINE = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']