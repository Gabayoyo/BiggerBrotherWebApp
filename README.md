# BiggerBrotherWebApp

A student project that extracts pose data from exercise videos and uses it to derive repetition metrics (reps, speed, durations) and estimate reps in reserve (RIR; proximity to failure) via personalised calibration data.  
The tool works both as a command-line interface (CLI) and through a Streamlit web app (WIP).

## What it does

- **Pose extraction** – MediaPipe Pose Landmarker detects 33 body landmarks per frame.
- **Angle & velocity calculation** – Various joint angles and absolute velocities are calculated from the frame-by-frame pose data.
- **Rep detection** – Peaks and troughs in the angle automatically identify repetitions.
- **Metric assembly** – Mean concentric speed, rep duration, and other higher-level metrics are calculated per rep.
- **RIR estimation** – Reps in reserve can be estimated when a calibration video is provided.
- **Velocity–load curve** – Polynomial regression is performed on the user's previous rep metrics, allowing for individualised fatigue benchmarking on subsequent sets.
  (Requires at least 3 reps.)
- **Visualisation** – Optional 2D skeleton animation and velocity‑load curve plot.

## Features

- Supports exercises like **bicep curl** and **tricep extension** (extendable via configuration).
- Caching of pose data for faster re‑runs (`--cache-data`).
- Choice of laterality: `bilateral`, `left`, or `right`.
- Built entirely in Python with well‑known libraries.

## Installation & Dependencies

Clone the repository and install the required packages:

```
python3 -m pip install -r requirements.txt
``` 

These packages include:
- `opencv-python`
- `numpy`
- `mediapipe`
- `ipywidgets`
- `pandas`
- `scipy`
- `tabulate`

Additional standard libraries like `matplotlib` are used and should be available in most Python environments.

The first run will automatically download the MediaPipe pose landmark model (models/pose_landmarker_full.task).

## Usage

(Web app is WIP)

Run the CLI from the project root:
```
python3 src/bigger_brother.py path/to/video.mp4 "bicep_curl" 10 --laterality left --cache-data
``` 
In this case, the video consists of a person curling 10kg just on their left arm. The metric data will also be cached locally.

**Note: It is important to ensure input videos have good lighting and that lift is not obscured by any foreign objects. Ideally, the lift should be performed and recorded perpendicular to the camera in order to make sure MediaPipe's pose estimation model best estimates joint positions.**

## Arguments

| Argument              | Type    | Description                                                   |
|-----------------------|---------|---------------------------------------------------------------|
| `input_path`          | path    | Video file to process                                         |
| `exercise`            | string  | Exercise name (e.g. `bicep_curl`, `tricep_extension`)         |
| `weight`              | float   | Weight used (kg)                                              |
| `--laterality`        | string  | `bilateral`, `left`, or `right` (default: `bilateral`)        |
| `--calibration-path`  | path    | Optional calibration video for RIR estimation                 |
| `--cache-dir`         | path    | Directory to store cached pose data (default: `./cache`)      |
| `--cache-data`        | flag    | Enable caching of pose outputs                                |
| `--visualise`         | flag    | Show skeleton animation after processing                      |
| `--visualise-curve`   | flag    | Plot the fitted velocity–load curve                           |

## Notes for developers

- Each frame must contain exactly 33 pose landmarks; otherwise an error is raised.
- The velocity–load curve fit needs at least 3 detected reps – shorter sets will be rejected
- Rep detection uses scipy.signal.find_peaks with a default prominence of 30.
This can be tweaked in src/analysis/compute_rep.py if needed.
- Pose data can be cached to a pickle file by passing --cache-data. This saves time when experimenting with different analysis parameters.
- The web app (Streamlit) interface is still a WIP