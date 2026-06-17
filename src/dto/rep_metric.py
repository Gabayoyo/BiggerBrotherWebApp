from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class RepMetric:
    """Structural info known as soon as a rep is detected."""
    rep_number: int
    ecc_start_frame: int
    ecc_end_frame: int
    con_start_frame: int
    con_end_frame: int
    rom_degrees: int
    con_duration_s: float
    rep_duration_s: float
    mean_concentric_speed_ms: Optional[float] = None
