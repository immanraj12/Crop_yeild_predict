import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page title
st.set_page_config(page_title="Crop Yield Prediction", layout="centered")

# Load the trained model
with open('crop_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler (if you used one)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Load the training columns
with open('X_columns.pkl', 'rb') as f:
    X_columns = pickle.load(f)

# Title of the app
st.title("🌱 Crop Yield Prediction App")

st.markdown("""
Enter the agricultural data below to predict the expected yield per hectare.
""")

# User inputs
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=50.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
days_to_harvest = st.number_input("Days to Harvest", min_value=1, value=100)

# Prediction button
if st.button("Predict"):
    # Prepare the input data
    input_data = pd.DataFrame([[rainfall, temperature, days_to_harvest]],
                              columns=['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest'])
    

    # Ensure the input data has the same columns as the training data
    input_data = input_data.reindex(columns=X_columns, fill_value=0)

    # Scale the data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Display the result
    st.success(f"🌾 Predicted Yield: {prediction[0]:.2f} tons per hectare")

    # Optional: Show more details
    st.write("### Input Data")
    st.write(input_data)
  