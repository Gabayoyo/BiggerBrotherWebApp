import sys
import os
from matplotlib.dates import FR
import numpy as np
import pytest

from dto.frame_data import Landmark, FrameData
from types import SimpleNamespace
from landmark_dicts import LANDMARK_INDICES

np.random.seed(0)

# tests the from_mediapipe class method of the Landmark class,
# ensuring it correctly converts mediapipe landmarks to Landmark instances
def test_from_mediapipe():
    """from_mediapipe should correctly convert a mediapipe landmark to a Landmark instance."""
    mp_lm = SimpleNamespace(x=0.1, y=0.2, z=-0.5, visibility=0.9, presence=1.0)
    result = Landmark.from_mediapipe(mp_lm)
    assert result.x == 0.1
    assert result.y == 0.2
    assert result.z == -0.5
    assert result.visibility == 0.9
    assert result.presence == 1.0

def test_get_landmark(make_landmarks):
    """get_landmark should return the correct landmark for every index."""
    landmarks = make_landmarks()
    frame_data = FrameData(frame_number=0, timestamp_s=0.0, landmarks=landmarks, world_landmarks=landmarks)
    for i in range(33):
        lm = frame_data.get_landmark(i)
        assert isinstance(lm, Landmark)
        assert lm.x == float(i)
        assert lm.y == float(i % 5)
        assert lm.z == float(i % 3)
        assert lm.visibility == 1.0
        assert lm.presence == 1.0

def test_get_landmark_by_name(make_landmarks):
    """get_landmark_by_name should return the correct landmark for every known name."""
    landmarks = make_landmarks()
    frame_data = FrameData(frame_number=0, timestamp_s=0.0, landmarks=landmarks, world_landmarks=landmarks)

    for name in LANDMARK_INDICES.keys():
        name_lm = frame_data.get_landmark_by_name(name)
        lm = frame_data.get_landmark(LANDMARK_INDICES[name])

        assert isinstance(lm, Landmark)
        assert isinstance(name_lm, Landmark)
        assert lm.x == name_lm.x
        assert lm.y == name_lm.y
        assert lm.z == name_lm.z
        assert lm.visibility == name_lm.visibility
        assert lm.presence == name_lm.presence

def test_get_world_landmark(make_landmarks):
    """get_world_landmark should return the correct world landmark for every index."""
    landmarks = make_landmarks(world_offset=(1.0, 2.0, 3.0))
    frame_data = FrameData(frame_number=0, timestamp_s=0.0, landmarks=landmarks, world_landmarks=landmarks)
    for i in range(33):
        lm = frame_data.get_world_landmark(i)
        assert isinstance(lm, Landmark)
        assert lm.x == float(i) + 1.0
        assert lm.y == float(i % 5) + 2.0
        assert lm.z == float(i % 3) + 3.0
        assert lm.visibility == 1.0
        assert lm.presence == 1.0

def test_get_world_landmark_by_name(make_landmarks):
    """get_world_landmark_by_name should return the correct world landmark for every known name."""
    landmarks = make_landmarks()
    # use different world landmarks with offset
    world = make_landmarks(world_offset=(10.0, 20.0, 30.0))
    fd = FrameData(frame_number=0, timestamp_s=0.0, landmarks=landmarks, world_landmarks=world)

    for name, idx in LANDMARK_INDICES.items():
        wlm = fd.get_world_landmark_by_name(name)
        expected = world[idx]
        assert isinstance(wlm, Landmark)
        assert wlm.x == expected.x
        assert wlm.y == expected.y
        assert wlm.z == expected.z
        assert wlm.visibility == expected.visibility
        assert wlm.presence == expected.presence


def test_get_world_landmark_by_name_invalid(make_landmarks):
    """get_world_landmark_by_name should raise KeyError for an invalid name."""
    landmarks = make_landmarks()
    fd = FrameData(frame_number=0, timestamp_s=0.0, landmarks=landmarks, world_landmarks=landmarks)
    with pytest.raises(KeyError):
        fd.get_world_landmark_by_name("NON_EXISTENT_LANDMARK")


def test_get_keypoints_array(make_landmarks):
    """get_keypoints_array should return a (33, 3) float array matching all landmarks."""
    landmarks = make_landmarks()
    fd = FrameData(frame_number=0, timestamp_s=0.0, landmarks=landmarks, world_landmarks=landmarks)
    kp = fd.get_keypoints_array()
    assert isinstance(kp, np.ndarray)
    assert kp.shape == (33, 3)
    for i, lm in enumerate(landmarks):
        assert np.allclose(kp[i], lm.to_array())


def test_get_visible_landmarks(make_landmarks):
    """get_visible_landmarks should return only landmarks with visibility > 0.7."""
    # create landmarks with mixed visibility
    lms = make_landmarks(visibility=0.5)  # all below threshold
    # manually set a few to high visibility
    lms[0] = Landmark(x=0, y=0, z=0, visibility=0.9, presence=1.0)
    lms[5] = Landmark(x=1, y=1, z=1, visibility=0.8, presence=1.0)
    lms[10] = Landmark(x=2, y=2, z=2, visibility=0.71, presence=1.0)  # exactly above 0.7? property uses >0.7 so 0.7 excluded
    fd = FrameData(frame_number=0, timestamp_s=0.0, landmarks=lms, world_landmarks=lms)
    visible = fd.get_visible_landmarks()
    # Only indices 0 and 5 should be >0.7 (0.8, 0.9). 0.71 > 0.7 → yes. But test with exact threshold.
    # Actually property visible is >0.7, so 0.71 qualifies. We'll keep it simple.
    visible_indices = [i for i, lm in visible]
    assert 0 in visible_indices
    assert 5 in visible_indices
    # Since other landmarks have visibility 0.5, they should not appear.
    assert len(visible) == 3  # 0,5,10 because 0.71 > 0.7
    # verify each returned tuple
    for i, lm in visible:
        assert lm.visibility > 0.7
        assert i < 33


def test_get_visible_landmarks_none(make_landmarks):
    """get_visible_landmarks should return empty list when no landmarks exceed threshold."""
    lms = make_landmarks(visibility=0.3)
    fd = FrameData(frame_number=0, timestamp_s=0.0, landmarks=lms, world_landmarks=lms)
    visible = fd.get_visible_landmarks()
    assert visible == []


def test_get_limb(make_landmarks):
    """get_limb should return the three world landmarks for a given tuple of indices."""
    landmarks = make_landmarks()
    world = make_landmarks(world_offset=(5.0, 10.0, 15.0))
    fd = FrameData(frame_number=0, timestamp_s=0.0, landmarks=landmarks, world_landmarks=world)

    # Use indices for a known limb, e.g., LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
    limb_indices = (11, 13, 15)  # typical MediaPipe indices, but must be in LANDMARK_INDICES
    # Better to get from landmark_dicts like in original tests
    from landmark_dicts import get_landmark_indices_from_exercise
    joints = get_landmark_indices_from_exercise("bicep_curl")[0]  # returns a tuple of 3 ints
    limb = fd.get_limb(joints)
    assert len(limb) == 3
    for i, idx in enumerate(joints):
        assert isinstance(limb[i], Landmark)
        # Should return world landmarks, not landmarks
        assert limb[i].x == world[idx].x
        assert limb[i].y == world[idx].y
        assert limb[i].z == world[idx].z


def test_get_lm_xyzv_by_name(make_landmarks):
    """get_lm_xyzv_by_name should return a numpy array of coordinates and the visibility."""
    landmarks = make_landmarks()
    world = make_landmarks(world_offset=(1.0, 2.0, 3.0))
    fd = FrameData(frame_number=0, timestamp_s=0.0, landmarks=landmarks, world_landmarks=world)

    for name in LANDMARK_INDICES.keys():
        xyz, vis = fd.get_lm_xyzv_by_name(name)
        wlm = world[LANDMARK_INDICES[name]]
        assert isinstance(xyz, np.ndarray)
        assert xyz.shape == (3,)
        assert np.allclose(xyz, wlm.to_array())
        assert vis == wlm.visibility


def test_get_lm_xyzv_by_name_invalid(make_landmarks):
    """get_lm_xyzv_by_name should raise KeyError for an unknown name."""
    landmarks = make_landmarks()
    fd = FrameData(frame_number=0, timestamp_s=0.0, landmarks=landmarks, world_landmarks=landmarks)
    with pytest.raises(KeyError):
        fd.get_lm_xyzv_by_name("INVALID")


def test_frame_data_from_mediapipe(make_landmarks):
    """FrameData.from_mediapipe should construct FrameData from fake mediapipe objects."""
    # Create a mock pose_landmarks with 33 fake landmarks
    class MockLandmark:
        def __init__(self, x, y, z, visibility, presence):
            self.x = x
            self.y = y
            self.z = z
            self.visibility = visibility
            self.presence = presence

    class MockPoseLandmarks:
        def __init__(self):
            self.landmark = [MockLandmark(i, i%2, i%3, 1.0, 1.0) for i in range(33)]

    pose = MockPoseLandmarks()
    world = [MockLandmark(i+0.1, (i%2)+0.2, (i%3)+0.3, 0.9, 0.8) for i in range(33)]  # mock world landmarks

    fd = FrameData.from_mediapipe(5, 0.5, pose, world)
    assert fd.frame_number == 5
    assert fd.timestamp_s == 0.5
    assert len(fd.landmarks) == 33
    # Check that the first landmark matches the mock
    assert fd.landmarks[0].x == 0.0
    assert fd.landmarks[0].y == 0.0
    assert fd.landmarks[0].z == 0.0
    assert fd.landmarks[0].visibility == 1.0
    # World landmarks are assigned directly from the passed list
    assert fd.world_landmarks is world  # assuming the method stores the list as-is


def test_frame_data_from_mediapipe_empty_pose():
    """FrameData.from_mediapipe should handle an empty pose landmarks list (unlikely but robust)."""
    class MockPoseLandmarks:
        landmark = []

    with pytest.raises(ValueError, match="Expected 33 landmarks"):
        FrameData.from_mediapipe(0, 0.0, MockPoseLandmarks(), [])