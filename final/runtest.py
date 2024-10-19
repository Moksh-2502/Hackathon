import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Load the saved model and scaler
scaler = joblib.load('scaler.joblib')
model = joblib.load('stacking_model.pkl')

# Create input fields in the sidebar
st.sidebar.header("User Input")

Pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, value=0)
Glucose = st.sidebar.number_input('Glucose', min_value=0.0, value=0.0)
BloodPressure = st.sidebar.number_input('Blood Pressure', min_value=0.0, value=0.0)
SkinThickness = st.sidebar.number_input('Skin Thickness', min_value=0.0, value=0.0)
Insulin = st.sidebar.number_input('Insulin', min_value=0.0, value=0.0)
BMI = st.sidebar.number_input('BMI', min_value=0.0, value=0.0)
DiabetesPedigreeFunction = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, value=0.0)
Age = st.sidebar.number_input('Age', min_value=0, value=0)

# Button to submit the input
if st.sidebar.button('Predict'):
    # Create DataFrame from user input
    user_data = pd.DataFrame({
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age]
    })

    # Handle zero values
    columns_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    user_data[columns_with_zero] = user_data[columns_with_zero].replace(0, np.nan)
    user_data.fillna(user_data.median(), inplace=True)

    # Scale the input
    user_data_scaled = scaler.transform(user_data)

    # Make prediction
    prediction = model.predict(user_data_scaled)
    probability = model.predict_proba(user_data_scaled)[:, 1]

    # Display the result
    if prediction[0] == 1:
        st.success(f"**Diabetes Risk: Positive** (Probability: {probability[0]:.2f})")
    else:
        st.success(f"**Diabetes Risk: Negative** (Probability: {probability[0]:.2f})")
