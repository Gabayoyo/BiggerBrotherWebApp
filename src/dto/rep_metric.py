from dataclasses import dataclass

@dataclass(frozen=True)
class RepBoundaries:
    """Structural info known as soon as a rep is detected."""
    rep_number: int
    eccentric_start_frame: int
    eccentric_end_frame: int
    concentric_start_frame: int
    concentric_end_frame: int
    rom_degrees: int
    rep_duration_s: float

@dataclass(frozen=True)
class RepMetrics:
    """Derived metrics, computed from RepBoundaries + frame data."""
    boundaries: RepBoundaries
    peak_concentric_speed_ms: float