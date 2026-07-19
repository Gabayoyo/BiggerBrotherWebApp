import numpy as np
import pytest

from bigger_brother import BiggerBrother
from dto.rep_metric import RepMetric

pytestmark = pytest.mark.e2e


@pytest.fixture
def bb_instance(tmp_path):
    cache = tmp_path / "cache"
    return BiggerBrother(cache_dir=cache, cache_data=False)


def _make_metrics(speeds: list[float]) -> list[RepMetric]:
    return [
        RepMetric(
            rep_number=i + 1,
            con_start_frame=i * 10,
            con_end_frame=i * 10 + 5,
            ecc_start_frame=i * 10 - 5,
            ecc_end_frame=i * 10,
            rom_start=30,
            rom_degrees=150,
            rep_duration_s=1.0,
            con_duration_s=0.5,
            mean_concentric_speed_ms=s,
        )
        for i, s in enumerate(speeds)
    ]


class TestEstimateRir:
    def test_happy_path(self, bb_instance):
        """Pure computation: target metrics + VL model → RiR int."""
        target_metrics = _make_metrics([1.8, 1.6, 1.5, 1.4])
        model = np.poly1d([0.1, 0.5, 10])
        rir = bb_instance.estimate_rir(target_metrics, model)
        assert isinstance(rir, int)
        assert rir >= 0

    def test_single_rep_returns_zero_or_positive(self, bb_instance):
        """A single rep with no fatigue context returns a plausible RiR."""
        target_metrics = _make_metrics([1.9])
        model = np.poly1d([0.1, 0.5, 10])
        rir = bb_instance.estimate_rir(target_metrics, model)
        assert isinstance(rir, int)
        assert rir >= 0

    def test_empty_metrics_raises(self, bb_instance):
        """Zero reps should raise."""
        model = np.poly1d([0.1, 0.5, 10])
        with pytest.raises(ValueError, match="at least one rep"):
            bb_instance.estimate_rir([], model)
