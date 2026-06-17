from dataclasses import dataclass
from pathlib import Path
from dto.rep_metric import RepMetric
from tabulate import tabulate

# returned by analyse_reps(). Contains the per-rep metrics as well as the exercise and video path
# does not contain RiR estimates
# designed solely as a DTO for the rep analysis endpoint
@dataclass
class RepAnalysisResult:

    video_path: Path
    # may want a more complex field type here if we want to include metadata about the exercise/video
    exercise: str
    metrics: list[RepMetric]

    # returns a string table representation of the analysis result for console output.
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
        table_width = len(table_str.splitlines()[0])
        title_line = title.center(table_width)
        return f"{title_line}\n{table_str}"

    def console_output(self) -> str:
        """Return a string representation of the analysis result for console output."""
        return (
            f"\n{self.summary_table(f'Analysis Results ({self.exercise})')}\n"
        )

# returned by estimate_rir(). Contains target metrics + RiR estimate
# option to return failure metrics as well, but not required for the endpoint
# designed solely as a DTO for the RiR estimation endpoint
@dataclass
class RirAnalysisResult:

    video_path: Path
    metrics: list[RepMetric]
    rir_estimate: int
    # rir_rationale: str
    failure_metrics: list[RepMetric] | None = None

    def summary_table(self) -> str:
        base = RepAnalysisResult(self.video_path, self.metrics).summary_table()
        return base + f"\nEstimated RiR: {self.rir_estimate} rep(s)"