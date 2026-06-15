import numpy as np
from scipy.signal import find_peaks
from typing import List

from dto.results import RepMetrics, RepBoundaries

def detect_reps(angles: List[float], fps: float,
                is_flexion: bool,
                prominence_deg: float = 5.0,
                min_phase_duration_s: float = 0.2) -> List[RepMetrics]:
    """
    Detect repetitions from a smoothed joint-angle sequence.

    Parameters
    ----------
    angles : list of float, length N
        Smoothed joint angle at each frame.
    fps : float
        Frames per second of the video.
    eccentric_decreasing : bool
        True if the angle decreases during the eccentric phase
        (e.g. squat, push-up), False if it increases (e.g. bicep curl).
    prominence_deg : float
        Minimum angular prominence (deg) for a reversal to be counted.
    min_phase_duration_s : float
        Minimum time (seconds) between direction changes to avoid jitter.

    Returns
    -------
    list of RepMetrics
        One entry per detected full repetition.
    """
    angles = np.asarray(angles, dtype=float)
    n = len(angles)
    if n < 3:
        return []

    # 1. Velocity (deg/s)
    velocity = np.gradient(angles, 1.0 / fps)

    # 2. Find extrema
    min_frames = int(min_phase_duration_s * fps)
    peaks, _ = find_peaks(angles, prominence=prominence_deg, distance=min_frames)
    valleys, _ = find_peaks(-angles, prominence=prominence_deg, distance=min_frames)

    # Combine and sort by frame index
    events = [(f, 'peak') for f in peaks] + [(f, 'valley') for f in valleys]
    events.sort(key=lambda x: x[0])

    # 3. Pair into reps
    rep_metrics = []
    rep_num = 0
    i = 0
    while i < len(events) - 2:
        f1, t1 = events[i]
        f2, t2 = events[i+1]
        f3, t3 = events[i+2]

        # Must alternate: t1 != t2 and t1 == t3
        if t1 == t2 or t1 != t3:
            i += 1
            continue

        # 4. Label phases according to is_flexion
        # Leg 1: f1 -> f2, Leg 2: f2 -> f3
        # Determine if leg1 is decreasing (peak -> valley) or increasing (valley -> peak)
        leg1_is_decreasing = (t1 == 'peak' and t2 == 'valley')

        if not is_flexion:
            if leg1_is_decreasing:
                ecc_start, ecc_end = f1, f2
                con_start, con_end = f2, f3
            else:
                # leg1 is increasing, so it becomes concentric, leg2 eccentric
                con_start, con_end = f1, f2
                ecc_start, ecc_end = f2, f3
        else:
            # eccentric is increasing
            if leg1_is_decreasing:
                con_start, con_end = f1, f2
                ecc_start, ecc_end = f2, f3
            else:
                ecc_start, ecc_end = f1, f2
                con_start, con_end = f2, f3

        # 5. Build metrics
        duration = (con_end - ecc_start) / fps
        rom = abs(angles[ecc_end] - angles[ecc_start])
        # peak concentric speed (absolute value)
        conc_vel = velocity[con_start: con_end+1]
        peak_speed = np.max(np.abs(conc_vel))

        boundaries = RepBoundaries(
            rep_number=rep_num + 1,
            eccentric_start_frame=int(ecc_start),
            eccentric_end_frame=int(ecc_end),
            concentric_start_frame=int(con_start),
            concentric_end_frame=int(con_end),
            rep_duration_s=round(duration, 4)
        )
        metrics = RepMetrics(
            boundaries=boundaries,
            rom_degrees=round(rom, 2),
            peak_concentric_speed_ms=round(peak_speed, 2)
        )
        rep_metrics.append(metrics)
        rep_num += 1

        # Move to next potential rep (E2 becomes the start of next rep if pattern continues)
        i += 2

    return rep_metrics