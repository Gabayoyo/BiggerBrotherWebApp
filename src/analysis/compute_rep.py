from scipy.signal import find_peaks

from dto.rep_metric import RepMetric

def compute_reps(
    angles: list[float],
    fps: float,
    is_flexion: bool = True,
    prominence: float = 30,
) -> list[RepMetric]:

    values = [a for a in angles if a is not None]
    frames = list(range(len(values)))

    peaks, _   = find_peaks( values, prominence=prominence)
    troughs, _ = find_peaks([-v for v in values], prominence=prominence)

    extrema = (
        [(frames[i], values[i], "peak")   for i in peaks] +
        [(frames[i], values[i], "trough") for i in troughs]
    )
    extrema.sort(key=lambda x: x[0])

    if not extrema:
        return []

    # Merge consecutive same-type extrema, keeping the more extreme value
    cleaned = [extrema[0]]
    for curr in extrema[1:]:
        prev = cleaned[-1]
        if curr[2] == prev[2]:
            if curr[2] == "peak":
                cleaned[-1] = curr if curr[1] > prev[1] else prev
            else:
                cleaned[-1] = curr if curr[1] < prev[1] else prev
        else:
            cleaned.append(curr)

    results: list[RepMetric] = []
    pending_trough = (frames[0], values[0], "trough")
    rep_number = 1

    for i, e in enumerate(cleaned):
        if e[2] == "trough":
            pending_trough = e

        elif e[2] == "peak" and pending_trough is not None:
            peak_frame, peak_val = e[0], e[1]

            next_trough = next(
                (cleaned[j] for j in range(i + 1, len(cleaned)) if cleaned[j][2] == "trough"),
                None,
            )

            trough_to_peak_frames = (pending_trough[0], peak_frame)
            peak_to_trough_frames = (peak_frame, next_trough[0]) if next_trough else None
            rom = round(peak_val)

            if is_flexion:
                # Flexion exercise (bicep curl, leg curl, etc.)
                # Concentric = peak → trough (angle decreasing)
                con_start = peak_frame
                con_end   = next_trough[0] if next_trough else None
                # Eccentric = trough → peak (angle increasing)
                ecc_start = pending_trough[0]
                ecc_end   = peak_frame
            else:
                # Extension exercise (tricep pushdown, leg extension, etc.)
                # Concentric = trough → peak (angle increasing)
                con_start = pending_trough[0]
                con_end   = peak_frame
                # Eccentric = peak → trough (angle decreasing)
                ecc_start = peak_frame
                ecc_end   = next_trough[0] if next_trough else None

            con_sec = (con_end - con_start) / fps
            ecc_sec = ((ecc_end - ecc_start) / fps) if ecc_start is not None and ecc_end is not None else 0.0

            metric = RepMetric(
                rep_number=rep_number,
                con_start_frame=con_start,
                con_end_frame=con_end,
                ecc_start_frame=ecc_start,
                ecc_end_frame=ecc_end,
                rom_degrees=rom,
                rep_duration_s=round(con_sec + ecc_sec, 3),
                con_duration_s=round(con_sec, 3),
                mean_concentric_speed_ms=None
            )

            rep_number += 1
            pending_trough = None
            results.append(metric)

    return results