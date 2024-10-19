import pickle 
import pandas as pd


# Load data
df = pd.read_csv(r'model_heartprediction\heart.csv')

# Prepare features and target
X = df.drop(columns=['HeartDisease'])

# Load the model from the pickle file
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Predict function that takes user input
def predict_user_input(model):
    user_input = []
    
    # Taking user input for features
    user_input.append(int(input("Enter Sex [0: female, 1: male]: "))) 
    user_input.append(int(input("Enter Age: ")))                       
    user_input.append(int(input("Enter Chest Pain Type [0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic]: ")))    
    user_input.append(int(input("Enter Resting Blood Pressure [mm Hg]: ")))     
    user_input.append(int(input("Enter Serum Cholesterol [mm/dl]: ")))           
    user_input.append(int(input("Enter Fasting Blood Sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]: "))) 
    user_input.append(int(input("Enter Resting ECG [0: Normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy]: ")))            
    user_input.append(int(input("Enter Maximum Heart Rate Achieved [between 60 and 202]]: "))) 
    user_input.append(int(input("Enter Exercise Angina (0 or 1): ")))    
    user_input.append(float(input("Old Peak: "))) 
    user_input.append(int(input("Enter Slope of the Peak Exercise ST Segment [0: upsloping, 1: flat, 2: downsloping]: ")))  
    
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input], columns=X.columns)
    
    # Make prediction
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        print("Predicted: Heart Disease")
    else:
        print("Predicted: No Heart Disease")


# Load the model and make prediction
loaded_model = load_model(r"model_heartprediction/ada_model.pkl")
predict_user_input(loaded_model)