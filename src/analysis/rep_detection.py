import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from src.dto.frame_data import FrameData
from src.dto.results import RepBoundaries

def count_reps(
    frame_data: list[FrameData],
    prominence: float = 30,
    min_rom: float = 30,
) -> RepBoundaries:

    signal = df["angle_smooth"].dropna()
    frames = signal.index.to_numpy()
    values = signal.values

    peaks, _ = find_peaks(values, prominence=prominence)
    troughs, _ = find_peaks(-values, prominence=prominence)

    extrema = (
        [(frames[i], values[i], "peak") for i in peaks] +
        [(frames[i], values[i], "trough") for i in troughs]
    )
    extrema.sort(key=lambda x: x[0])

    if not extrema:
        results.reps_df = pd.DataFrame()
        return results

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

    pending_trough = (frames[0], values[0], "trough")
    reps = []

    for i in range(len(cleaned)):

        e = cleaned[i]

        if e[2] == "trough":
            pending_trough = e

        elif e[2] == "peak" and pending_trough is not None:

            peak_frame = e[0]
            peak_val = e[1]
            con_rom = abs(peak_val - pending_trough[1])

            if con_rom < min_rom:
                continue

            rep = {
                "con_start_frame": pending_trough[0],
                "con_end_frame": peak_frame,
                "con_frames": peak_frame - pending_trough[0],
                "con_sec": round((peak_frame - pending_trough[0]) / results.fps, 3),
                "rom_deg": round(con_rom, 1),
            }

            next_trough = next(
                (cleaned[j] for j in range(i + 1, len(cleaned)) if cleaned[j][2] == "trough"),
                None
            )

            if next_trough is not None:
                ecc_rom = abs(peak_val - next_trough[1])

                if ecc_rom >= min_rom:
                    rep["ecc_start_frame"] = peak_frame
                    rep["ecc_end_frame"] = next_trough[0]
                    rep["ecc_frames"] = next_trough[0] - peak_frame
                    rep["ecc_sec"] = round((next_trough[0] - peak_frame) / results.fps, 3)
                    rep["total_sec"] = round(rep["con_sec"] + rep["ecc_sec"], 3)
                    rep["ce_ratio"] = round(rep["ecc_sec"] / rep["con_sec"], 2)
                    rep["tempo"] = f"{rep['con_sec']}s / {rep['ecc_sec']}s"

            reps.append(rep)
            pending_trough = None

    rep_df = pd.DataFrame(reps)

    if not rep_df.empty:
        rep_df.index = rep_df.index + 1
        rep_df.index.name = "rep"

    results.reps = rep_df
    return results