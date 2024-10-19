import pandas as pd
import pickle 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score


# Load data
df = pd.read_csv('model_heartprediction\heart.csv')

# Prepare features and target
X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

# Encode categorical features
encoder = LabelEncoder()
X['Sex'] = encoder.fit_transform(X['Sex'])
X['ChestPainType'] = encoder.fit_transform(X['ChestPainType'])
X['RestingECG'] = encoder.fit_transform(X['RestingECG'])
X['ExerciseAngina'] = encoder.fit_transform(X['ExerciseAngina'])
X['ST_Slope'] = encoder.fit_transform(X['ST_Slope'])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)




lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

print("Training Accuracy of lr: ",lr_model.score(X_train, y_train))

y_pred = lr_model.predict(X_test)

print("Test Accuracy of lr: ",accuracy_score(y_test, y_pred))


rf_model = RandomForestClassifier(n_estimators=200, min_impurity_decrease=0.001)

rf_model.fit(X_train, y_train)

print("Training Accuracy of rf: ", rf_model.score(X_train, y_train))

y_pred = rf_model.predict(X_test)

print("Test Accuracy of rf: ", accuracy_score(y_test, y_pred))  


ada = AdaBoostClassifier(estimator=RandomForestClassifier(n_estimators=300, min_impurity_decrease=0.001), n_estimators=30)

ada.fit(X_train, y_train)

print("Training Accuracy of ada: ",ada.score(X_train, y_train))
print("Test Accuracy of ada: ",ada.score(X_test, y_test))



bagg = BaggingClassifier(estimator=RandomForestClassifier(n_estimators=200, min_impurity_decrease=0.001), 
                         n_estimators=100)

bagg.fit(X_train, y_train)

print("Training Accuracy of bagc: ",bagg.score(X_train, y_train))

y_pred = bagg.predict(X_test)

print("Test Accuracy of bagc: ",accuracy_score(y_test, y_pred))



xgb = XGBClassifier(n_estimators=1000, learning_rate=0.001, verbosity=0)

xgb.fit(X_train, y_train)

print("Training Accuracy of rf: ",xgb.score(X_train, y_train))

print("Test Accuracy of rf: ",xgb.score(X_test, y_test))

print("From the accuracies we can see that ada perform the best")

print("Classification Report for AdaBoost Model:")
print(classification_report(y_test, y_pred))

# Save the AdaBoost model to a pickle file
with open('ada_model.pkl', 'wb') as file:
    pickle.dump(ada, file)

print("AdaBoost model saved to ada_model.pkl")
