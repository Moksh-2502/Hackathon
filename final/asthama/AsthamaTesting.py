# Step 1: Import necessary libraries
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Step 2: Load the saved model and feature columns
model = joblib.load(r'asthama\respiratory_disease_model.pkl')
feature_columns = joblib.load(r'asthama\feature_columns.pkl')

print("Model and feature columns loaded successfully.")

# Step 3: Define a function to preprocess new input data
def preprocess_input(data_dict):
    """
    Preprocess the user input data to match the training data format.
    
    Parameters:
    data_dict (dict): Dictionary containing user inputs for each feature.
    
    Returns:
    pd.DataFrame: A DataFrame with a single row of preprocessed features.
    """
    # Convert the input dictionary to a DataFrame
    input_df = pd.DataFrame([data_dict])
    
    # Handle missing values (example: fill missing BMI with the mean)
    input_df['BMI'].fillna(input_df['BMI'].mean(), inplace=True)
    
    # Apply one-hot encoding for categorical variables to match training columns
    input_df = pd.get_dummies(input_df, columns=['Gender'], drop_first=True)
    
    # Add missing columns that were in the training data but are not in the input data
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing columns with default value 0
    
    # Ensure the columns are in the same order as the training data
    input_df = input_df[feature_columns]
    
    return input_df

# Step 4: Define a function to predict asthma based on user input
def predict_asthma(data_dict):
    """
    Predict whether a patient has asthma based on input features.
    
    Parameters:
    data_dict (dict): Dictionary containing user inputs for each feature.
    
    Returns:
    str: Prediction result (e.g., "Asthma" or "Not Asthma").
    """
    # Preprocess the input data
    preprocessed_data = preprocess_input(data_dict)
    
    # Make prediction using the loaded model
    prediction = model.predict(preprocessed_data)
    
    # Interpret the prediction
    result = "Asthma" if prediction[0] == 1 else "Not Asthma"
    return result

# Step 5: Take user input through the console for key features only
print("Please provide the following information for asthma diagnosis:")

user_input = {
    'Age': int(input("Enter Age: ")),
    'Gender': input("Enter Gender (Male/Female): "),
    'BMI': float(input("Enter BMI: ")),
    'Smoking': int(input("Enter Smoking status (0 = Non-Smoker, 1 = Smoker): ")),
    'FamilyHistoryAsthma': int(input("Enter Family History of Asthma (0 = No, 1 = Yes): ")),
    'LungFunctionFEV1': float(input("Enter Lung Function FEV1 value: ")),
    'LungFunctionFVC': float(input("Enter Lung Function FVC value: ")),
    'Wheezing': int(input("Enter Wheezing status (0 = No, 1 = Yes): ")),
    'ShortnessOfBreath': int(input("Enter Shortness of Breath status (0 = No, 1 = Yes): "))
}

# Step 6: Make a prediction with the user input
result = predict_asthma(user_input)
print(f"\nPrediction based on the input data: {result}")
