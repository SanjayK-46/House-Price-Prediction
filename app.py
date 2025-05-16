import streamlit as st
import pandas as pd
import joblib

# Load model components
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Exchange rate USD to INR (update as needed)
usd_to_inr = 82

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏡 House Price Prediction")

st.markdown("Enter the property details below:")

Beds = st.number_input("Number of Bedrooms", min_value=0, value=3)
Baths = st.number_input("Number of Bathrooms", min_value=0, value=2)
Sqft_home = st.number_input("Size of Home (in sqft)", min_value=0, value=1500)
Sqft_lot = st.number_input("Size of Lot (in sqft)", min_value=0, value=3000)
Build_year = st.number_input("Year Built", min_value=1800, max_value=2025, value=2010)

Type = st.text_input("Type (e.g., Single Family)", value="Single Family")

if st.button("Predict House Price"):
    Age = 2025 - Build_year
    input_data = {
        'Beds': Beds,
        'Baths': Baths,
        'Sqft_home': Sqft_home,
        'Sqft_lot': Sqft_lot,
        'Build_year': Build_year,
        'Age': Age,
        f'Type_{Type}': 1,
    }

    input_vector = {col: input_data.get(col, 0) for col in feature_columns}
    input_df = pd.DataFrame([input_vector])
    input_scaled = scaler.transform(input_df)
    prediction_usd = model.predict(input_scaled)[0]

    prediction_inr = prediction_usd * usd_to_inr

    st.success(f"Estimated Sale Price: ₹{prediction_inr:,.2f} INR")
