import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)
DEFAULT_MODEL_PATH = Path("models/pose_landmarker_full.task")


# download the pose landmarker if it isn't already present
def ensure_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        print(f"Downloading pose landmarker model to {model_path} …")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Download complete.")
    return model_path
