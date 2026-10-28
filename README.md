# Churn Prediction Service

Production-style MLE starter project for training, evaluating, and serving a customer churn model.

## Architecture

```text
synthetic/customer data -> train.py -> model artifact + metrics
                                      -> FastAPI /predict
                                      -> Docker container
```

The default dataset is deterministic and generated locally so the project is reproducible without credentials or external data downloads. Replace `src/churn_service/data.py` with a governed data connector for a production deployment.

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.churn_service.train
uvicorn src.churn_service.api:app --reload
```

Then open `http://127.0.0.1:8000/docs`.

## Docker

```powershell
docker build -t churn-prediction-service .
docker run --rm -p 8000:8000 churn-prediction-service
```

## Testing

```powershell
pytest
```

The training command writes artifacts to `artifacts/`, which is intentionally gitignored.
