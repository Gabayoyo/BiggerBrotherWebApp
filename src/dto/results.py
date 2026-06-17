from dataclasses import dataclass
from pathlib import Path
from dto.rep_metric import RepMetric
from tabulate import tabulate
    
@dataclass
class RepAnalysisResult:
    """Returned by analyze_reps(). Contains per-rep metrics only."""
    video_path: Path
    # May want a more complex field type here if we want to include metadata about the exercise/video
    exercise: str
    metrics: list[RepMetric]

    def summary_table(self, title: str = "Analysis Results") -> str:
        headers = ["Rep", "ROM°", "Mean Concentric Speed m/s", "Concentric Duration (s)", "Total Duration (s)"]
        table_data = []
        for m in self.metrics:
            table_data.append([
                m.rep_number,
                f"{m.rom_degrees:.1f}",
                f"{m.mean_concentric_speed_ms:.2f}",
                f"{m.con_duration_s:.2f}" if m.con_duration_s is not None else "N/A",
                f"{m.rep_duration_s:.2f}",
            ])
        table_str = tabulate(table_data, headers=headers, tablefmt="simple")
        # Centre the title over the table
        table_width = len(table_str.splitlines()[0])   # width of the separator line
        title_line = title.center(table_width)
        return f"{title_line}\n{table_str}"

    def console_output(self) -> str:
        """Return a string representation of the analysis result for console output."""
        return (
            f"\n{self.summary_table(f'Analysis Results ({self.exercise})')}\n"
        )

# needs duplication of RepMetrics as both are intended to be exclusive
# as return to their respective functions/pipelines/endpoints.
# Can refactor later if we want to decouple them
@dataclass
class RirAnalysisResult:
    """Returned by estimate_rir(). Contains target metrics + RiR estimate."""
    video_path: Path
    metrics: list[RepMetric]          # rep metrics of the target video
    rir_estimate: int
    # rir_rationale: str
    # Optionally include failure video metrics if you want to display them
    failure_metrics: list[RepMetric] | None = None

    def summary_table(self) -> str:
        base = RepAnalysisResult(self.video_path, self.metrics).summary_table()
        return base + f"\nEstimated RiR: {self.rir_estimate} rep(s)"
    
# probably a dto for metrics attached to a video file