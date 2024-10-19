import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE  # Correct import
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load and preprocess the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=columns)

# Replace 0 values with NaN for specific columns
columns_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns_with_zero] = data[columns_with_zero].replace(0, np.nan)

# Fill missing values with the median of each column
data.fillna(data.median(), inplace=True)

# Advanced Feature Engineering
data['Glucose_to_Insulin_Ratio'] = data['Glucose'] / (data['Insulin'] + 1)
data['BMI_Category'] = pd.cut(data['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])
data['Age_Category'] = pd.cut(data['Age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3])
data['Glucose_BP_Product'] = data['Glucose'] * data['BloodPressure']
data['BMI_Age_Interaction'] = data['BMI'] * data['Age']

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['BMI_Category', 'Age_Category'])

# Split the dataset
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Address class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Feature Selection
selector = SelectFromModel(estimator=RandomForestClassifier(n_estimators=100, random_state=42))
selector.fit(X_train_resampled, y_train_resampled)
X_train_selected = selector.transform(X_train_resampled)
X_test_selected = selector.transform(X_test_scaled)

# Hyperparameter tuning using Optuna
def objective(trial):
    xgb_params = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
    }

    model = xgb.XGBClassifier(**xgb_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
    score = cross_val_score(model, X_train_selected, y_train_resampled, cv=5, scoring='accuracy').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params

# Train models with optimized parameters
xgb_model = xgb.XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
rf_model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
lr_model = LogisticRegression(C=0.1, penalty='l2', random_state=42)
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

models = [xgb_model, rf_model, gb_model, lr_model, nn_model]
model_names = ['XGBoost', 'Random Forest', 'Gradient Boosting', 'Logistic Regression', 'Neural Network']

for model, name in zip(models, model_names):
    model.fit(X_train_selected, y_train_resampled)

# Stacking Ensemble
from sklearn.ensemble import StackingClassifier

stacking_model = StackingClassifier(
    estimators=[('xgb', xgb_model), ('rf', rf_model), ('gb', gb_model), ('lr', lr_model), ('nn', nn_model)],
    final_estimator=LogisticRegression(),
    cv=5
)

stacking_model.fit(X_train_selected, y_train_resampled)

# Save the models
joblib.dump(stacking_model, 'stacking_model.pkl')
joblib.dump(scaler, 'scaler.joblib')

print("Models saved successfully!")
