from pydantic import BaseModel


class RepMetricSchema(BaseModel):
    rep_number: int
    ecc_start_frame: int
    ecc_end_frame: int
    con_start_frame: int
    con_end_frame: int
    rom_start: int
    rom_degrees: int
    con_duration_s: float
    rep_duration_s: float
    mean_concentric_speed_ms: float | None = None


class AnalyseResponse(BaseModel):
    metrics: list[RepMetricSchema]
    estimated_1rm: float


class CalibrateResponse(BaseModel):
    coefficients: list[float]


class EstimateRirRequest(BaseModel):
    metrics: list[RepMetricSchema]
    coefficients: list[float]


class EstimateRirResponse(BaseModel):
    rir_estimate: int
