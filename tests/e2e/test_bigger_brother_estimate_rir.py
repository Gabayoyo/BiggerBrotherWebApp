from unittest import mock

import numpy as np
import pytest

from bigger_brother import BiggerBrother
from dto.results import RepMetric, RirAnalysisResult

pytestmark = pytest.mark.e2e


@pytest.fixture
def target_video(tmp_path):
    return tmp_path / "target.mp4"


@pytest.fixture
def calib_video(tmp_path):
    return tmp_path / "calib.mp4"


@pytest.fixture
def bb_instance(tmp_path):
    cache = tmp_path / "cache"
    return BiggerBrother(cache_dir=cache, cache_data=False)


def _make_mock_metrics(num=4):
    """Create a list of RepMetric with decaying speeds (valid for VL curve)."""
    metrics = []
    for i in range(num):
        speed = 2.0 - i * 0.2
        metrics.append(
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
                mean_concentric_speed_ms=speed,
            )
        )
    return metrics


class TestEstimateRir:
    def test_happy_path(self, bb_instance, target_video, calib_video, input_config):
        """Mock video processing and compute_vl_curve to return a known RIR value."""
        mock_metrics = _make_mock_metrics()
        dummy_rir = 3

        with (
            mock.patch.object(
                bb_instance.pose_estimator, "process_video", return_value=([], 10.0)
            ),
            mock.patch(
                "bigger_brother.compute_metrics", return_value=mock_metrics
            ) as mock_cm,
            mock.patch(
                "bigger_brother.compute_vl_curve",
                return_value=np.poly1d([0.1, 0.5, 10]),
            ) as mock_vl,
            mock.patch(
                "bigger_brother.estimate_rir_from_curve", return_value=dummy_rir
            ) as mock_rir,
        ):
            result = bb_instance.estimate_rir(target_video, calib_video, input_config)

        assert isinstance(result, RirAnalysisResult)
        assert result.rir_estimate == dummy_rir

        # Verify compute_metrics was called for calibration video
        mock_cm.assert_called_once()
        # compute_vl_curve called with the mock metrics
        mock_vl.assert_called_once_with(mock_metrics, visualise_curve=False)
        # estimate_rir_from_curve called with mock metrics and the polynomial
        mock_rir.assert_called_once_with(mock_metrics, mock_vl.return_value)

    def test_insufficient_reps_raises(
        self, bb_instance, target_video, calib_video, input_config
    ):
        """When calibration yields <3 reps, compute_vl_curve raises ValueError."""
        few_metrics = _make_mock_metrics(2)  # only 2 reps

        with (
            mock.patch.object(
                bb_instance.pose_estimator, "process_video", return_value=([], 10.0)
            ),
            mock.patch("bigger_brother.compute_metrics", return_value=few_metrics),
            pytest.raises(ValueError, match="has only 2 rep"),
        ):
            bb_instance.estimate_rir(target_video, calib_video, input_config)

    def test_visualise_flag_passes_through(
        self, bb_instance, target_video, calib_video, input_config, monkeypatch
    ):
        """Visualise flags are forwarded to compute_metrics and compute_vl_curve."""
        monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
        input_config.visualise = True
        input_config.visualise_curve = True
        mock_metrics = _make_mock_metrics()

        with (
            mock.patch.object(
                bb_instance.pose_estimator, "process_video", return_value=([], 10.0)
            ),
            mock.patch(
                "bigger_brother.compute_metrics", return_value=mock_metrics
            ) as mock_cm,
            mock.patch(
                "bigger_brother.compute_vl_curve", return_value=np.poly1d([1])
            ) as mock_vl,
            mock.patch("bigger_brother.estimate_rir_from_curve", return_value=0),
        ):
            bb_instance.estimate_rir(target_video, calib_video, input_config)

        # Check that visualise flags were passed
        mock_cm.assert_called_once()
        _, kwargs = mock_cm.call_args
        assert kwargs.get("visualise") is True
        mock_vl.assert_called_once()
        _, vl_kwargs = mock_vl.call_args
        assert vl_kwargs.get("visualise_curve") is True
