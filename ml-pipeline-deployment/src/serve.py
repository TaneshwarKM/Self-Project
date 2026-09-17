"""
serve.py
FastAPI service exposing the trained churn model as a REST endpoint.
Validates input with Pydantic, loads the model once at startup (not
per-request), and returns both the prediction and the probability so
downstream consumers can apply their own decision threshold.
Run with: uvicorn src.serve:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "models/churn_model.joblib"

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        bundle = joblib.load(MODEL_PATH)
        _state["model"] = bundle["model"]
        _state["feature_columns"] = bundle["feature_columns"]
    except FileNotFoundError:
        _state["model"] = None
        _state["feature_columns"] = None
    yield
    _state.clear()


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


class ChurnRequest(BaseModel):
    tenure_months: float = Field(..., ge=0, le=600)
    monthly_charges: float = Field(..., ge=0)
    support_tickets_last_90d: int = Field(..., ge=0)
    num_products: int = Field(..., ge=1, le=20)
    avg_login_frequency_per_week: float = Field(..., ge=0, le=14)


class ChurnResponse(BaseModel):
    churn_prediction: int
    churn_probability: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _state.get("model") is not None}


@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run train.py first.")

    feature_columns = _state["feature_columns"]
    features = pd.DataFrame([[getattr(request, col) for col in feature_columns]], columns=feature_columns)
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return ChurnResponse(churn_prediction=prediction, churn_probability=round(probability, 4))
