import pandas as pd
import numpy as np
from joblib import load
import webbrowser


def get_user_symptoms(symptom_list):
    symptoms = {symptom: 0 for symptom in symptom_list}
    
    print("Select the symptoms you are experiencing from the list below:")
    for i, symptom in enumerate(symptom_list):
        print(f"{i + 1}. {symptom.replace('_', ' ')}")  
    
    selected = input("\nEnter the numbers corresponding to your symptoms, separated by commas (e.g., 1, 3, 5): ")
    selected_indices = [int(x.strip()) - 1 for x in selected.split(',') if x.strip().isdigit()]
    
    for index in selected_indices:
        if 0 <= index < len(symptom_list):
            symptoms[symptom_list[index]] = 1
    
    return symptoms




def advanced_diagnosis(disease):
    disease_links = {
        'Bronchial Asthma': "https://colab.research.google.com/drive/14N8dcLXgRWXu-6R3B8po0oXIPVSkeh-y?usp=sharing",
        'Heart Attack': "https://colab.research.google.com/drive/1uI8Uog2tnMFXN72w21nRVtWX4y0BVGXy?usp=sharing",
        'Diabetes': "https://colab.research.google.com/drive/1KGQ1O8JwAfgo8e5FZPkw3DVJQirhWmbk?usp=sharing",
        'Allergy': "https://colab.research.google.com/drive/14gZH_Fkx6HAOj5K-GKEcAAlekagF6BKr?usp=sharing",
    }
    
    if disease in disease_links:
        colab_link = disease_links[disease]
        print(f"\nRedirecting to advanced diagnosis for {disease}...")
        webbrowser.open(colab_link)
    else:
        print(f"No advanced diagnosis available for {disease}.")


def offer_advanced_diagnosis(disease):
    if disease in ['Bronchial Asthma', 'Heart Attack', 'Diabetes', 'Allergy']:
        user_input = input(f"\nThe predicted disease is {disease}. Would you like to proceed with advanced diagnosis? (yes/no): ").strip().lower()
        
        if user_input == 'yes':
            advanced_diagnosis(disease)
        else:
            print(f"Advanced diagnosis for {disease} skipped.")
    else:
        print(f"\nNo advanced diagnosis available for {disease}.")


if __name__ == '__main__':
    symptom_list = [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 
        'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 
        'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 
        'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 
        'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 
        'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 
        'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 
        'dehydration', 'indigestion', 'headache', 'yellowish_skin', 
        'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 
        'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 
        'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 
        'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 
        'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 
        'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 
        'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 
        'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 
        'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 
        'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 
        'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 
        'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 
        'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 
        'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 
        'movement_stiffness', 'spinning_movements', 'loss_of_balance', 
        'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 
        'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 
        'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 
        'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 
        'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 
        'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 
        'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 
        'lack_of_concentration', 'visual_disturbances', 
        'receiving_blood_transfusion', 'receiving_unsterile_injections', 
        'coma', 'stomach_bleeding', 'distention_of_abdomen', 
        'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 
        'prominent_veins_on_calf', 'palpitations', 'painful_walking', 
        'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 
        'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 
        'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
    ]
    
    user_symptoms = get_user_symptoms(symptom_list)

    df_test = pd.DataFrame(columns=symptom_list)
    df_test.loc[0] = np.array(list(user_symptoms.values()))
    clf = load("final\Disease-Prediction-from-Symptoms-master\saved_model\decision_tree.joblib")
    result = clf.predict(df_test)[0]
    
    print(f"\nPredicted disease: {result}")

    offer_advanced_diagnosis(result)