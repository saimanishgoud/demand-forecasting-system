from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Create FastAPI app
app = FastAPI(title="Demand Forecasting API")

# Load trained model
model = joblib.load("models/best_model.pkl")

# Input schema
class SalesInput(BaseModel):
    Store: int
    DayOfWeek: int
    Customers: int
    Open: int
    Promo: int
    SchoolHoliday: int
    CompetitionDistance: float

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Demand Forecasting API is Running!"
    }

# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "Healthy"
    }

# Prediction endpoint
@app.post("/predict")
def predict(data: SalesInput):

    input_data = pd.DataFrame({
        "Store": [data.Store],
        "DayOfWeek": [data.DayOfWeek],
        "Customers": [data.Customers],
        "Open": [data.Open],
        "Promo": [data.Promo],
        "SchoolHoliday": [data.SchoolHoliday],
        "CompetitionDistance": [data.CompetitionDistance]
    })

    prediction = model.predict(input_data)

    return {
        "Predicted Sales": round(float(prediction[0]), 2)
    }