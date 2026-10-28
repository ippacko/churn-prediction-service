from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import make_dataset

LOGGER = logging.getLogger(__name__)
FEATURES = ["tenure_months", "monthly_charges", "support_tickets", "contract_months"]


def train(output_dir: Path = Path("artifacts")) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = make_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data["churn"], test_size=0.2, random_state=42, stratify=data["churn"]
    )
    model = Pipeline(
        [
            ("features", ColumnTransformer([("numeric", StandardScaler(), FEATURES)])),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("trained model metrics=%s", metrics)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
