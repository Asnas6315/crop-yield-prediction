import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

# Load model and columns
model = joblib.load("crop_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Title
st.title("🌾 Crop Yield Prediction System")
st.markdown("Predict agricultural crop yield based on environmental and farming inputs.")

# Sidebar Inputs
st.sidebar.header("Input Parameters")

crop_year = st.sidebar.number_input("Crop Year", min_value=1997, max_value=2030, value=2018)
area = st.sidebar.number_input("Area (in hectares)", min_value=0.0, value=5000.0)
rainfall = st.sidebar.number_input("Annual Rainfall (mm)", min_value=0.0, value=1200.0)
fertilizer = st.sidebar.number_input("Fertilizer (kg)", min_value=0.0, value=20000.0)
pesticide = st.sidebar.number_input("Pesticide (kg)", min_value=0.0, value=500.0)

# Prediction Button
if st.sidebar.button("Predict Yield"):

    input_dict = {
        'Crop_Year': crop_year,
        'Area': area,
        'Annual_Rainfall': rainfall,
        'Fertilizer': fertilizer,
        'Pesticide': pesticide
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)

    st.success(f"🌱 Predicted Yield: {round(prediction[0], 2)} Tons/Hectare")

# Footer
st.markdown("---")
st.markdown("Developed using Machine Learning & Streamlit")