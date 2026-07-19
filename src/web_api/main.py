import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from bigger_brother import BiggerBrother
from dto.input_config import InputConfig
from dto.rep_metric import RepMetric
from web_api.schemas import (
    AnalyseResponse,
    CalibrateResponse,
    EstimateRirRequest,
    EstimateRirResponse,
    RepMetricSchema,
)

app = FastAPI(title="BiggerBrotherWebApp")

# Singleton — no caching for temp uploads
_bb: BiggerBrother | None = None


def get_bb() -> BiggerBrother:
    global _bb
    if _bb is None:
        _bb = BiggerBrother(cache_data=False)
    return _bb


def _save_to_temp(video: UploadFile) -> Path:
    """Write uploaded video bytes to a named temp file and return the path."""
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = video.file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
    return tmp_path


def _metric_to_schema(m: RepMetric) -> RepMetricSchema:
    return RepMetricSchema(
        rep_number=m.rep_number,
        ecc_start_frame=m.ecc_start_frame,
        ecc_end_frame=m.ecc_end_frame,
        con_start_frame=m.con_start_frame,
        con_end_frame=m.con_end_frame,
        rom_start=m.rom_start,
        rom_degrees=m.rom_degrees,
        con_duration_s=m.con_duration_s,
        rep_duration_s=m.rep_duration_s,
        mean_concentric_speed_ms=m.mean_concentric_speed_ms,
    )


def _schema_to_metric(s: RepMetricSchema) -> RepMetric:
    return RepMetric(
        rep_number=s.rep_number,
        ecc_start_frame=s.ecc_start_frame,
        ecc_end_frame=s.ecc_end_frame,
        con_start_frame=s.con_start_frame,
        con_end_frame=s.con_end_frame,
        rom_start=s.rom_start,
        rom_degrees=s.rom_degrees,
        con_duration_s=s.con_duration_s,
        rep_duration_s=s.rep_duration_s,
        mean_concentric_speed_ms=s.mean_concentric_speed_ms,
    )


@app.post("/analyse", response_model=AnalyseResponse)
async def analyse(
    video: UploadFile = File(...),  # noqa: B008
    exercise: str = Form(...),  # noqa: B008
    weight: float = Form(...),  # noqa: B008
    laterality: str = Form("bilateral"),  # noqa: B008
):
    """Upload a video → per-rep metrics + estimated 1RM.

    State stays on the client: store the returned ``metrics`` in JS memory.
    """
    tmp_path = _save_to_temp(video)
    try:
        bb = get_bb()
        config = InputConfig(
            exercise=exercise,
            weight=weight,
            laterality=laterality,
            visualise=False,
        )
        result = bb.get_metrics(tmp_path, config)
        return AnalyseResponse(
            metrics=[_metric_to_schema(m) for m in result.metrics],
            estimated_1rm=result.estimated_1rm or 0.0,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/calibrate", response_model=CalibrateResponse)
async def calibrate(
    video: UploadFile = File(...),  # noqa: B008
    exercise: str = Form(...),  # noqa: B008
    weight: float = Form(...),  # noqa: B008
    laterality: str = Form("bilateral"),  # noqa: B008
):
    """Upload a calibration video → VL curve coefficients.

    Store the returned ``coefficients`` in JS memory.
    """
    tmp_path = _save_to_temp(video)
    try:
        bb = get_bb()
        config = InputConfig(
            exercise=exercise,
            weight=weight,
            laterality=laterality,
            visualise=False,
        )
        result = bb.get_metrics(tmp_path, config)
        model = bb.compute_vl_curve(result.metrics)
        return CalibrateResponse(coefficients=model.coefficients.tolist())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/estimate-rir", response_model=EstimateRirResponse)
async def estimate_rir(body: EstimateRirRequest):
    """Given previously-stored metrics + calibration coefficients, estimate RiR.

    The client should hold ``metrics`` from ``/analyse`` and
    ``coefficients`` from ``/calibrate`` in JS memory.
    """
    try:
        metrics = [_schema_to_metric(m) for m in body.metrics]
        model = np.poly1d(body.coefficients)
        bb = get_bb()
        rir = bb.estimate_rir(metrics, model)
        return EstimateRirResponse(rir_estimate=rir)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
