from fastapi import FastAPI
from pydantic import BaseModel
from src.serving.inference import predict

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Production-ready ML API for predicting customer churn",
    version="1.0.0"
)


@app.get("/")
def health():
    return {
        "status": "healthy",
        "message": "Telco Churn API is running"
    }


class CustomerData(BaseModel):
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def predict_customer(data: CustomerData):
    result = predict(data.model_dump())
    return {
        "prediction": result
    }