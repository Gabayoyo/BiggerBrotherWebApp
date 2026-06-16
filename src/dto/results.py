from dataclasses import dataclass
from pathlib import Path
from dto.rep_metric import RepMetrics
    
@dataclass
class RepAnalysisResult:
    """Returned by analyze_reps(). Contains per-rep metrics only."""
    video_path: Path
    # May want a more complex field type here if we want to include metadata about the exercise/video
    exercise: str
    metrics: list[RepMetrics]

    def summary_table(self) -> str:
        """Return a formatted string table of per-rep metrics."""
        header = f"{'Rep':>4} {'ROM°':>8} {'Con m/s':>8} {'Dur(s)':>7}"
        rows = [header, "-" * len(header)]
        for m in self.metrics:
            rows.append(
                f"{m.boundaries.rep_number:>4} "
                f"{m.boundaries.rom_degrees:>8.1f} "
                f"{m.peak_concentric_speed_ms:>8.2f} "
                f"{m.boundaries.rep_duration_s:>7.2f}"
            )
        return "\n".join(rows)

# needs duplication of RepMetrics as both are intended to be exclusive
# as return to their respective functions/pipelines/endpoints.
# Can refactor later if we want to decouple them
@dataclass
class RirAnalysisResult:
    """Returned by estimate_rir(). Contains target metrics + RiR estimate."""
    video_path: Path
    metrics: list[RepMetrics]          # rep metrics of the target video
    rir_estimate: int
    # rir_rationale: str
    # Optionally include failure video metrics if you want to display them
    failure_metrics: list[RepMetrics] | None = None

    def summary_table(self) -> str:
        base = RepAnalysisResult(self.video_path, self.metrics).summary_table()
        return base + f"\nEstimated RiR: {self.rir_estimate} rep(s)"
    
# probably a dto for metrics attached to a video file