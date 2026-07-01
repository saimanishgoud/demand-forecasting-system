import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/best_model.pkl")

# Create sample input
sample_data = pd.DataFrame({
    "Store": [2],
    "DayOfWeek": [5],
    "Customers": [650],
    "Open": [1],
    "Promo": [1],
    "SchoolHoliday": [0],
    "CompetitionDistance": [1270]
})

# Make prediction
prediction = model.predict(sample_data)

# Display result
print("=" * 40)
print("Demand Forecast Prediction")
print("=" * 40)
print(f"Predicted Sales: {prediction[0]:.2f}")