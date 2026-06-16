from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class RepMetric:
    """Structural info known as soon as a rep is detected."""
    rep_number: int
    eccentric_start_frame: int
    eccentric_end_frame: int
    concentric_start_frame: int
    concentric_end_frame: int
    rom_degrees: int
    rep_duration_s: float
    peak_concentric_speed_ms: Optional[float] = None