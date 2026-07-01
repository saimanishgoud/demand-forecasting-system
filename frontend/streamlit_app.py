import streamlit as st
import requests

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Demand Forecasting System",
    page_icon="📈",
    layout="wide"
)

# -------------------- Sidebar --------------------
st.sidebar.title("📋 Navigation")
page = st.sidebar.selectbox(
    "Choose a Page",
    ["Prediction", "Model Performance", "About Project"]
)

# =====================================================
# Prediction Page
# =====================================================
if page == "Prediction":

    st.title("📈 Demand Forecasting System")
    st.write("Predict store sales using the trained Random Forest model.")

    col1, col2 = st.columns(2)

    with col1:
        store = st.number_input("Store ID", min_value=1, value=1)

        day = st.selectbox(
            "Day of Week",
            [1, 2, 3, 4, 5, 6, 7]
        )

        customers = st.number_input(
            "Customers",
            min_value=0,
            value=500
        )

        competition = st.number_input(
            "Competition Distance",
            min_value=0.0,
            value=1270.0
        )

    with col2:
        open_store = st.selectbox(
            "Store Open",
            [1, 0]
        )

        promo = st.selectbox(
            "Promotion",
            [1, 0]
        )

        school = st.selectbox(
            "School Holiday",
            [0, 1]
        )

    st.write("")

    if st.button("Predict Sales"):

        data = {
            "Store": int(store),
            "DayOfWeek": int(day),
            "Customers": int(customers),
            "Open": int(open_store),
            "Promo": int(promo),
            "SchoolHoliday": int(school),
            "CompetitionDistance": float(competition)
        }

        try:

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=data
            )

            if response.status_code == 200:

                result = response.json()

                st.success("Prediction Successful ✅")

                st.metric(
                    "Predicted Sales",
                    f"₹ {result['Predicted Sales']:.2f}"
                )

            else:

                st.error(f"API Error: {response.status_code}")
                st.write(response.text)

        except Exception as e:

            st.error("Could not connect to FastAPI Server")
            st.write(e)

# =====================================================
# Model Performance
# =====================================================
elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.subheader("Best Model: Random Forest")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", "319.45")
    col2.metric("RMSE", "531.03")
    col3.metric("R² Score", "0.9809")

    st.write("---")

    st.subheader("Model Comparison")

    comparison = {
        "Linear Regression": 0.8538,
        "Random Forest": 0.9809,
        "XGBoost": 0.9413
    }

    st.bar_chart(comparison)

# =====================================================
# About Project
# =====================================================
else:

    st.title("ℹ️ About Project")

    st.markdown("""
### Demand Forecasting System

This project predicts future store sales using Machine Learning.

### Dataset
Rossmann Store Sales Dataset

### Algorithms
- Linear Regression
- Random Forest
- XGBoost

### Best Model
Random Forest

### Technologies
- Python
- Pandas
- Scikit-Learn
- FastAPI
- Streamlit

### Objective
Help businesses forecast demand and improve inventory planning.
""")

st.write("---")
st.caption("Developed by Sai Manish Goud | Demand Forecasting System")