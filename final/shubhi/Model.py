import zipfile
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib


# Define the path to your zip file and the extraction directory
zip_file_path = r'C:\Users\jidub\OneDrive\Documents\hackathon\Datathon\Final Model\shubhi\archive '  
extraction_dir = r'C:\Users\jidub\OneDrive\Documents\hackathon\Datathon\Final Model\shubhi'  

# Step 1: Unzip the dataset
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_dir)

# Verify the contents
print(os.listdir(extraction_dir))

# Define paths to training and testing directories
train_dir = os.path.join(extraction_dir, 'train_set')
test_dir = os.path.join(extraction_dir, 'test_set')

# Step 2: Load Images using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important to keep labels in order
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important to keep labels in order
)

# Step 3: Extract features using a pre-trained model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Freeze the base model layers

# Create a new model that outputs features from the base model
model = Model(inputs=base_model.input, outputs=base_model.output)

# Extract features for the entire training set
features_train = model.predict(train_generator)
features_test = model.predict(test_generator)

# Step 4: Flatten the features
features_train = features_train.reshape((features_train.shape[0], -1))
features_test = features_test.reshape((features_test.shape[0], -1))

# Get labels from the generator
train_labels = train_generator.classes
test_labels = test_generator.classes

# Step 5: Standardize the features
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# Step 6: Train SVM on the extracted features
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(features_train, train_labels)

# Step 7: Evaluate SVM on the test data
accuracy = svm_model.score(features_test, test_labels)
print(f"SVM Accuracy: {accuracy}")

# Calculate the training accuracy
training_accuracy = svm_model.score(features_train, train_labels)
print(f"SVM Training Accuracy: {training_accuracy * 100:.2f}%")

# Load the saved SVM model
svm_model_loaded = joblib.load('svm_model.joblib')

# Now you can use svm_model_loaded to make predictions or evaluate

