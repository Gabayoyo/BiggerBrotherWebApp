import numpy as np
import pytest

from analysis.compute_vl_curve import compute_vl_curve


class TestComputeVLCurve:
    def test_successful_fit(self, make_metrics):
        """A valid set: peak early, enough post-peak reps."""
        speeds = [2.0, 1.8, 1.6, 1.5]  # peak at index 0
        metrics = make_metrics(speeds)
        poly = compute_vl_curve(metrics, visualise_curve=False)
        assert isinstance(poly, np.poly1d)
        # At 0% velocity loss the polynomial should predict the first rep_pct
        assert poly(0) == pytest.approx(25.0, abs=1.0)  # rep 1 of 4 => 25%

    def test_too_few_reps_raises(self, make_metrics):
        metrics = make_metrics([1.0, 1.1])  # only 2 reps
        with pytest.raises(ValueError, match="has only 2 rep"):
            compute_vl_curve(metrics)

    def test_peak_too_late_raises(self, make_metrics):
        # 5 reps, peak at rep 3 (index 2) -> 3/5 = 60% > 33.4%
        speeds = [1.0, 1.1, 1.2, 1.1, 1.0]
        metrics = make_metrics(speeds)
        with pytest.raises(ValueError, match="past the.*threshold"):
            compute_vl_curve(metrics)

    def test_none_speeds_are_skipped(self, make_metrics):
        """Reps with None speed are ignored; remaining set must still be valid."""
        speeds = [2.0, None, 1.8, None, 1.6]  # after filtering -> [2.0, 1.8, 1.6]
        metrics = make_metrics(speeds)
        poly = compute_vl_curve(metrics, visualise_curve=False)
        # n after filtering = 3, peak at index 0 → first rep_pct = 1/3 * 100 ≈ 33.33
        assert poly(0) == pytest.approx(33.33, abs=0.5)

    def test_poly_degree_and_evaluation(self, make_metrics):
        """Check that the polynomial is quadratic and predicts both ends."""
        speeds = [2.2, 2.0, 1.8, 1.7, 1.6]  # peak index 0, 5 reps
        metrics = make_metrics(speeds)
        poly = compute_vl_curve(metrics, visualise_curve=False)
        coeffs = poly.coefficients
        assert len(coeffs) == 3  # quadratic
        # At 0% VL -> about 20% of set (rep 1 of 5)
        assert poly(0) == pytest.approx(20.0, abs=1.0)
        # At max VL -> about 100% of set
        v_ref = max(speeds)
        v_last = speeds[-1]
        max_vl = (v_ref - v_last) / v_ref * 100
        assert poly(max_vl) == pytest.approx(100.0, abs=1.0)

    def test_exact_three_reps_edge(self, make_metrics):
        """3 reps, peak at first rep -> 33.33% is just below threshold, allowed."""
        speeds = [2.0, 1.9, 1.8]
        metrics = make_metrics(speeds)
        poly = compute_vl_curve(metrics, visualise_curve=False)
        assert isinstance(poly, np.poly1d)

    def test_visualise_flag_runs_without_error(self, make_metrics, monkeypatch):
        """Ensure visualise_curve=True does not crash; suppress plot display."""
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda: None)
        speeds = [2.0, 1.8, 1.6, 1.5]
        metrics = make_metrics(speeds)
        # Should execute without exception
        poly = compute_vl_curve(metrics, visualise_curve=True)
        assert isinstance(poly, np.poly1d)
