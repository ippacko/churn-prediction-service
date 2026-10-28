from fastapi.testclient import TestClient

from src.churn_service.api import app


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction():
    response = TestClient(app).post(
        "/predict",
        json={
            "tenure_months": 3,
            "monthly_charges": 120,
            "support_tickets": 4,
            "contract_months": 1,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert isinstance(body["will_churn"], bool)


def test_rejects_invalid_customer():
    response = TestClient(app).post("/predict", json={"tenure_months": 0})
    assert response.status_code == 422
