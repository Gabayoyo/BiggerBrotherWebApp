from dataclasses import dataclass


# metrics for a single rep, returned by compute_metrics()
@dataclass(frozen=True)
class RepMetric:
    rep_number: int
    ecc_start_frame: int  # ecc = eccentric (muscle lengthening phase)
    ecc_end_frame: int
    con_start_frame: (
        int  # con = concentric (muscle shortening phase) and main phase of interest
    )
    con_end_frame: int
    rom_start: int  # where the rep was started
    rom_degrees: int  # range of motion in degrees
    con_duration_s: float  # duration of concentric phase in seconds
    rep_duration_s: float  # total duration of rep in seconds (concentric + eccentric)

    # mean concentric speed in meters per second, optional as it is calculated separately after rep detection
    mean_concentric_speed_ms: float | None = None
