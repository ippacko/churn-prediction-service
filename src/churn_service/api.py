from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .train import FEATURES, train

LOGGER = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).parents[2] / "artifacts" / "model.joblib"
app = FastAPI(title="Churn Prediction API", version="1.0.0")


class Customer(BaseModel):
    tenure_months: int = Field(ge=1, le=240)
    monthly_charges: float = Field(gt=0, le=1000)
    support_tickets: int = Field(ge=0, le=100)
    contract_months: int = Field(ge=1, le=120)


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool


def load_model():
    if not MODEL_PATH.exists():
        train(MODEL_PATH.parent)
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    try:
        model = load_model()
        frame = pd.DataFrame([customer.model_dump()], columns=FEATURES)
        probability = float(model.predict_proba(frame)[0, 1])
        return Prediction(churn_probability=round(probability, 4), will_churn=probability >= 0.5)
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.exception("prediction failed")
        raise HTTPException(status_code=500, detail="Model prediction failed") from exc
