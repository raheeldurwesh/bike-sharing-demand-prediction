import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load trained model
model = joblib.load("bike_sharing_best_model.pkl")

st.set_page_config(
    page_title="Bike Sharing Demand Prediction",
    page_icon="🚲",
    layout="centered"
)

st.title("🚲 Bike Sharing Demand Prediction")
st.write("Predict the number of bikes required based on weather and time.")

st.subheader("Enter Prediction Details")

col1, col2 = st.columns(2)

with col1:
    date = st.date_input("Date", value=datetime.today())
    hour = st.slider("Hour of Day", 0, 23, 8)
    season = st.selectbox(
    "Season",
    options=["springer", "summer", "fall", "winter"],
    format_func=lambda x: {
        "springer": "Spring",
        "summer": "Summer",
        "fall": "Autumn",
        "winter": "Winter"
    }[x]
    )
    weather = st.selectbox(
        "Weather",
        options=["Clear", "Mist", "Light Snow", "Heavy Rain"]
    )

with col2:
    temperature = st.slider("Temperature (normalized)", 0.0, 1.0, 0.5)
    feels_like = st.slider("Feels Like Temperature (normalized)", 0.0, 1.0, 0.5)
    humidity = st.slider("Humidity (normalized)", 0.0, 1.0, 0.5)
    windspeed = st.slider("Windspeed (normalized)", 0.0, 1.0, 0.2)

year = date.year
month = date.month
day = date.day
weekday = date.weekday()

holiday = st.selectbox("Is it a holiday?", ["No", "Yes"])

workingday = st.selectbox(
    "Is it a working day?",
    ["No Work", "Working Day"]
)

is_weekend = int(weekday >= 5)

# Match the feature names expected by the model
input_data = pd.DataFrame([{
    "season": season,
    "yr": int(year >= 2012),
    "mnth": month,
    "hr": hour,
    "holiday": holiday,
    "weekday": weekday,
    "workingday": workingday,
    "weathersit": weather,
    "temp": temperature,
    "atemp": feels_like,
    "hum": humidity,
    "windspeed": windspeed,
    "day": day,
    "is_weekend": is_weekend
}])

if st.button("Predict Bike Demand", type="primary"):
    try:
        prediction = model.predict(input_data)[0]
        prediction = max(0, int(round(prediction)))

        st.success(f"🚲 Predicted Bike Demand: {prediction} bikes")

        st.metric("Estimated Bikes Required", f"{prediction:,}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")