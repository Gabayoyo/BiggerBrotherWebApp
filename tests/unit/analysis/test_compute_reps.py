import pytest

from analysis.compute_reps import compute_reps
from dto.rep_metric import RepMetric


def _rep_metric(
    rep_number,
    con_start,
    con_end,
    ecc_start,
    ecc_end,
    rom_start,
    rom_deg,
    rep_dur,
    con_dur,
):
    """Create a RepMetric with given durations and frame indices, speed=None."""
    return RepMetric(
        rep_number=rep_number,
        con_start_frame=con_start,
        con_end_frame=con_end,
        ecc_start_frame=ecc_start,
        ecc_end_frame=ecc_end,
        rom_start=rom_start,
        rom_degrees=rom_deg,
        rep_duration_s=round(rep_dur, 3),
        con_duration_s=round(con_dur, 3),
        mean_concentric_speed_ms=None,
    )


class TestComputeRepsFlexion:
    def test_happy_path_two_reps(self):
        """Two full flexion reps: trough -> peak -> trough -> peak -> trough."""
        angles = [50, 80, 120, 160, 140, 100, 60, 90, 130, 110, 70]
        fps = 10.0
        expected = [
            _rep_metric(1, 3, 6, 0, 3, 50, 160, 0.6, 0.3),
            _rep_metric(2, 8, 10, 6, 8, 60, 130, 0.4, 0.2),
        ]
        reps = compute_reps(angles, fps, is_flexion=True, prominence=30)
        assert len(reps) == len(expected)
        for r, e in zip(reps, expected, strict=True):
            assert r.rep_number == e.rep_number
            assert r.con_start_frame == e.con_start_frame
            assert r.con_end_frame == e.con_end_frame
            assert r.ecc_start_frame == e.ecc_start_frame
            assert r.ecc_end_frame == e.ecc_end_frame
            assert r.rom_start == pytest.approx(e.rom_start)
            assert r.rom_degrees == e.rom_degrees
            assert r.rep_duration_s == pytest.approx(e.rep_duration_s)
            assert r.con_duration_s == pytest.approx(e.con_duration_s)
            assert r.mean_concentric_speed_ms is None

    def test_prominence_filters_small_wiggles(self):
        """Higher prominence ignores small oscillations that are not genuine peaks."""
        angles = [40, 100, 95, 100, 95, 100, 40]
        fps = 10.0
        reps_low = compute_reps(angles, fps, is_flexion=True, prominence=10)
        reps_high = compute_reps(angles, fps, is_flexion=True, prominence=30)
        assert len(reps_high) <= len(reps_low)

    def test_peak_merging_consecutive_peaks(self):
        """Consecutive peaks without an intervening trough are merged, keeping the highest."""
        angles = [50, 110, 111, 110, 50]
        fps = 10.0
        reps = compute_reps(angles, fps, is_flexion=True, prominence=10)
        assert len(reps) == 1
        assert reps[0].con_start_frame == 2
        assert reps[0].con_end_frame == 4
        assert reps[0].ecc_start_frame == 0
        assert reps[0].ecc_end_frame == 2
        assert reps[0].rom_degrees == 111

    def test_no_peaks_or_troughs_returns_empty(self):
        """Flat signal yields no reps."""
        angles = [100] * 20
        assert compute_reps(angles, 10, is_flexion=True) == []

    def test_only_troughs_no_peaks(self):
        """Signal with only a trough but no peak above prominence threshold returns empty."""
        angles = [100, 80, 60, 80, 100]
        reps = compute_reps(angles, 10, is_flexion=True, prominence=5)
        assert reps == []

    def test_start_with_peak_no_initial_trough(self):
        """Data starting at a peak: first frame used as initial trough, still produces reps."""
        angles = [150, 120, 100, 80, 100, 120, 150, 120, 100]
        reps = compute_reps(angles, 10, is_flexion=True, prominence=10)
        assert len(reps) >= 1


class TestComputeRepsExtension:
    def test_happy_path_extension(self):
        """Two full extension reps: trough -> peak -> trough -> peak -> trough."""
        angles = [50, 80, 120, 160, 140, 100, 60, 90, 130, 110, 70]
        fps = 10.0
        expected = [
            _rep_metric(1, 0, 3, 3, 6, 50, 160, 0.6, 0.3),
            _rep_metric(2, 6, 8, 8, 10, 60, 130, 0.4, 0.2),
        ]
        reps = compute_reps(angles, fps, is_flexion=False, prominence=30)
        assert len(reps) == len(expected)
        for r, e in zip(reps, expected, strict=True):
            assert r.rep_number == e.rep_number
            assert r.con_start_frame == e.con_start_frame
            assert r.con_end_frame == e.con_end_frame
            assert r.ecc_start_frame == e.ecc_start_frame
            assert r.ecc_end_frame == e.ecc_end_frame
            assert r.rom_start == pytest.approx(e.rom_start)
            assert r.rom_degrees == e.rom_degrees
            assert r.rep_duration_s == pytest.approx(e.rep_duration_s)
            assert r.con_duration_s == pytest.approx(e.con_duration_s)


class TestEdgeCases:
    def test_all_nones(self):
        """All None values yield empty result."""
        angles = [None, None, None]
        assert compute_reps(angles, 10) == []

    def test_mixed_nones(self):
        """None values are filtered; rep detection works on remaining valid angles."""
        angles = [None, 50, None, 100, None, 60, None]
        fps = 5
        reps = compute_reps(angles, fps, is_flexion=True, prominence=10)
        assert len(reps) == 1
        assert reps[0].rep_number == 1
        assert reps[0].rom_degrees == 100

    def test_fps_affects_durations(self):
        """Phase durations scale inversely with fps."""
        angles = [50, 100, 50]
        reps_10fps = compute_reps(angles, 10, is_flexion=True, prominence=10)
        reps_20fps = compute_reps(angles, 20, is_flexion=True, prominence=10)
        assert len(reps_10fps) == 1
        assert len(reps_20fps) == 1
        assert reps_20fps[0].con_duration_s == pytest.approx(0.05)
        assert reps_10fps[0].con_duration_s == pytest.approx(0.1)

    def test_single_rep(self):
        """One complete rep from trough to peak back to trough."""
        angles = [30, 90, 30]
        reps = compute_reps(angles, 10, is_flexion=True, prominence=10)
        assert len(reps) == 1
        assert reps[0].rep_number == 1
        assert reps[0].rom_degrees == 90
        assert reps[0].rom_start == 30

    def test_missing_next_trough_flexion(self):
        """When no trough follows the last peak, the concentric phase ends at the last frame."""
        # Peak at index 3 (value 140), then drops to 120 – no trough after.
        angles = [60, 80, 100, 140, 120]  # fps=10
        reps = compute_reps(angles, 10, is_flexion=True, prominence=10)
        assert len(reps) == 1
        # Last frame index is 4, and con_end should be that frame.
        assert reps[0].con_start_frame == 3
        assert reps[0].con_end_frame == 4
        # Duration is (4 - 3) / 10 = 0.1
        assert reps[0].con_duration_s == pytest.approx(0.1)
        # Eccentric: trough at index 0 (60) to peak at 3
        assert reps[0].ecc_start_frame == 0
        assert reps[0].ecc_end_frame == 3

    def test_missing_next_trough_extension(self):
        """Extension: when no trough follows the last peak, eccentric phase ends at the last frame."""
        # Peak at index 3 (value 150), then drops to 140 – no trough after.
        angles = [50, 70, 90, 150, 140]  # fps=10
        reps = compute_reps(angles, 10, is_flexion=False, prominence=10)
        assert len(reps) == 1
        # Last frame index is 4
        assert reps[0].con_start_frame == 0  # trough → peak
        assert reps[0].con_end_frame == 3
        assert reps[0].ecc_start_frame == 3
        assert reps[0].ecc_end_frame == 4  # fallback to last frame
        # Eccentric duration (4 - 3) / 10 = 0.1
        assert reps[0].rep_duration_s - reps[0].con_duration_s == pytest.approx(0.1)
