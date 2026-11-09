#!/usr/bin/env python3
"""
scripts/router_service.py 

Local http service for Orion's gradiant boosted action router.

Run:
  .venv-router/bin/uvicorn scripts.router_service:app --host 127.0.0.1 --port 8082
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from scripts.guardrails import apply_guardrails

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "action_router.joblib"

FEATURES = [
    "age",
    "max_hr",
    "baseline_5km_s",
    "resting_hr",
    "rhr_delta",
    "sleep_min",
    "sleep_efficiency",
    "deep_sleep_min",
    "rem_sleep_min",
    "wake_min",
    "acwr",
    "acute_load_7d",
    "chronic_load_28d",
    "today_load",
    "readiness",
    "fatigue",
    "soreness",
    "stress",
    "mood",
    "injury_reported",
]

class AthleteState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: Optional[float] = None
    max_hr: Optional[float] = None
    baseline_5km_s: Optional[float] = None
    resting_hr: Optional[float] = None
    rhr_delta: Optional[float] = None
    sleep_min: Optional[float] = None
    sleep_efficiency: Optional[float] = None
    deep_sleep_min: Optional[float] = None
    rem_sleep_min: Optional[float] = None
    wake_min: Optional[float] = None
    acwr: Optional[float] = None
    acute_load_7d: Optional[float] = None
    chronic_load_28d: Optional[float] = None
    today_load: Optional[float] = None
    readiness: Optional[float] = None
    fatigue: Optional[float] = None
    soreness: Optional[float] = None
    stress: Optional[float] = None
    mood: Optional[float] = None
    injury_reported: Optional[int] = Field(default=None, ge=0, le=1)

class RouteResponse(BaseModel):
    action: str
    router_action: str
    confidence: float
    probabilities: dict[str, float]
    guardrail_override: bool = False
    guardrail_reason: str | None = None
    missing_features: list[str] = []

model = None
feature_names: list[str] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_names

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Router model not found at {MODEL_PATH}. "
            "Run `python3 scripts/train_action_router.py` first."
        )
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_names = artifact["features"]

    if feature_names != FEATURES:
        raise RuntimeError(
            "Feature order mismatch between training script and service. "
            f"Expected {FEATURES}, got {feature_names}"
        )
    yield

    model = None
    feature_names = []

app = FastAPI(title="Orion Action Router", version="1.0", lifespan=lifespan)
@app.get("/health")
def health():
    return{
            "status": "ok",
            "model_path": str(MODEL_PATH),
            "features": len(feature_names),
    }


@app.post("/route", response_model=RouteResponse)
def route(state: AthleteState):
    if model is None:
        raise HTTPException(status_code=503, detail="Router model is not loaded")

    values = []
    missing = []

    for name in feature_names:
        value = getattr(state, name)
        if value is None:
            missing.append(name)
            values.append(np.nan)
        else:
            values.append(float(value))

    row = np.array([values], dtype=float)
    probabilities = model.predict_proba(row)[0]
    classes = model.classes_

    probability_map = {
        str(label): float(probability)
        for label, probability in zip(classes, probabilities)
    }
    best_index = int(np.argmax(probabilities))
    router_action = str(classes[best_index])
    guardrail = apply_guardrails(state.model_dump(), router_action)

    return RouteResponse(
        action=guardrail.action,
        router_action=router_action,
        confidence=float(probabilities[best_index]),
        probabilities=probability_map,
        guardrail_override=guardrail.overridden,
        guardrail_reason=guardrail.reason,
        missing_features=missing,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("scripts.router_service:app", host="127.0.0.1", port=8082)
