FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
RUN python -m src.churn_service.train
EXPOSE 8000
CMD ["uvicorn", "src.churn_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
