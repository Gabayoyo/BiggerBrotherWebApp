from dataclasses import dataclass
from pathlib import Path
from tabulate import tabulate
from typing import Optional

from dto.rep_metric import RepMetric

def _summary_table( title: str, metrics: list[RepMetric], estimated_1rm: Optional[float] = None) -> str:
        headers = [
            "Rep",
            "ROM°",
            "Mean Concentric Speed m/s",
            "Concentric Duration (s)",
            "Total Duration (s)",
        ]
        table_data = []
        for m in metrics:
            table_data.append(
                [
                    m.rep_number,
                    f"{m.rom_degrees:.1f}",
                    f"{m.mean_concentric_speed_ms:.2f}",
                    f"{m.con_duration_s:.2f}"
                    if m.con_duration_s is not None
                    else "N/A",
                    f"{m.rep_duration_s:.2f}",
                ]
            )
        table_str = tabulate(table_data, headers=headers, tablefmt="simple")
        table_width = len(table_str.splitlines()[0])
        title_line = title.center(table_width)
        table = f"{title_line}\n{table_str}"
        output = f"{table}\n\nEstimated 1rm: {estimated_1rm:.2f} kg"
        return output


# returned by analyse_reps(). Contains the per-rep metrics as well as the exercise and video path
# does not contain RiR estimates
# designed solely as a DTO for the rep analysis endpoint
@dataclass
class RepAnalysisResult:
    video_path: Path
    # may want a more complex field type here if we want to include metadata about the exercise/video
    exercise: str
    metrics: list[RepMetric]
    estimated_1rm: Optional[float] = None

    # returns a string table representation of the analysis result for console output.
    def summary_table(self, title: str = "Analysis Results") -> str:
        headers = [
            "Rep",
            "ROM°",
            "Mean Concentric Speed m/s",
            "Concentric Duration (s)",
            "Total Duration (s)",
        ]
        table_data = []
        for m in self.metrics:
            table_data.append(
                [
                    m.rep_number,
                    f"{m.rom_degrees:.1f}",
                    f"{m.mean_concentric_speed_ms:.2f}",
                    f"{m.con_duration_s:.2f}"
                    if m.con_duration_s is not None
                    else "N/A",
                    f"{m.rep_duration_s:.2f}",
                ]
            )
        table_str = tabulate(table_data, headers=headers, tablefmt="simple")
        table_width = len(table_str.splitlines()[0])
        title_line = title.center(table_width)
        table = f"{title_line}\n{table_str}"
        output = f"{table}\n\nEstimated 1rm: {self.estimated_1rm:.2f} kg"
        return output

    def console_output(self) -> str:
        return f"{_summary_table(f'Analysis Results ({self.exercise})', metrics=self.metrics, estimated_1rm=self.estimated_1rm)}\n"


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
    estimated_1rm: Optional[float] = None

    def summary_table(self) -> str:
        base = _summary_table(
            f"RiR Estimation Results ({self.video_path.name})",
            metrics=self.metrics,
            estimated_1rm=self.estimated_1rm,
        )
        return base + f"\nEstimated RiR: {self.rir_estimate} rep(s)"
