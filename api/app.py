from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import random

app = FastAPI(title="Demand Forecasting API")

MODEL_PATH = "models/best_model.pkl"
model = None

# Load model safely
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading failed: {e}")
        model = None
else:
    print("WARNING: Model file not found. Running in demo mode.")

class SalesInput(BaseModel):
    Store: int
    DayOfWeek: int
    Customers: int
    Open: int
    Promo: int
    SchoolHoliday: int
    CompetitionDistance: float

@app.get("/")
def home():
    return {"message": "Demand Forecasting API is Running!"}

@app.get("/health")
def health():
    return {"status": "Healthy"}

@app.post("/predict")
def predict(data: SalesInput):

    # Create input dataframe
    input_data = pd.DataFrame([{
        "Store": data.Store,
        "DayOfWeek": data.DayOfWeek,
        "Customers": data.Customers,
        "Open": data.Open,
        "Promo": data.Promo,
        "SchoolHoliday": data.SchoolHoliday,
        "CompetitionDistance": data.CompetitionDistance
    }])

    # If model exists → real prediction
    if model is not None:
        prediction = model.predict(input_data)
        return {
            "Predicted Sales": round(float(prediction[0]), 2),
            "mode": "ml"
        }

    # Otherwise → demo mode
    return {
        "Predicted Sales": round(random.uniform(1000, 5000), 2),
        "mode": "demo"
    }
