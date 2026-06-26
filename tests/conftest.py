import sys
import os
import pytest

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(TEST_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


import numpy as np
from dto.frame_data import Landmark

@pytest.fixture(scope="function")
def make_landmarks():
    """
    Factory fixture that returns a function to build a list of 33 Landmark objects
    with optional overrides for visibility and world_offset.
    """
    def _factory(visibility=1.0, world_offset=(0.0, 0.0, 0.0)):
        lms = []
        for i in range(33):
            x = float(i) + world_offset[0]
            y = float(i % 5) + world_offset[1]
            z = float(i % 3) + world_offset[2]
            lms.append(Landmark(x=x, y=y, z=z, visibility=visibility, presence=1.0))
        return lms
    return _factory