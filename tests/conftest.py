import os
import sys

import pytest

from dto.frame_data import Landmark
from dto.input_config import InputConfig
from dto.rep_metric import RepMetric

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(TEST_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


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


@pytest.fixture(scope="function")
def make_metrics():
    """Quick factory for RepMetric instances."""

    def _factory(speeds: list, rom_degrees=90, con_dur=0.5, rep_dur=1.0):
        metrics = []
        for i, spd in enumerate(speeds, start=1):
            metrics.append(
                RepMetric(
                    rep_number=i,
                    ecc_start_frame=i * 100,
                    ecc_end_frame=i * 100 + 40,
                    con_start_frame=i * 100 + 41,
                    con_end_frame=i * 100 + 80,
                    rom_start=0,
                    rom_degrees=rom_degrees,
                    con_duration_s=con_dur,
                    rep_duration_s=rep_dur,
                    mean_concentric_speed_ms=spd,
                )
            )
        return metrics

    return _factory


@pytest.fixture
def input_config():
    """Default InputConfig for tests – can be overridden per test."""
    return InputConfig(
        exercise="bicep_curl",
        weight=10.0,
        laterality="right",
        visualise=False,
        visualise_curve=False,
    )
