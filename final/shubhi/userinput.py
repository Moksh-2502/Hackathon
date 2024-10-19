import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Function to upload images (using Tkinter for file dialog)
def upload_image():
    Tk().withdraw()  # Hide the root window
    img_path = askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if img_path:
        print(f'Uploaded image: {img_path}')
        return img_path
    else:
        print("No image selected.")
        return None

# Function to preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Resize image
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale to [0, 1]
    return img_array

# Function to extract features from the uploaded image
def extract_features(img_array):
    features = model.predict(img_array)  # Use the VGG16 model to extract features
    features = features.reshape((features.shape[0], -1))  # Flatten features
    return scaler.transform(features)  # Standardize features

# Function to diagnose allergy from the uploaded image
def diagnose_allergy(img_path):
    img_array = preprocess_image(img_path)  # Preprocess the image
    features = extract_features(img_array)  # Extract features
    prediction = svm_model.predict(features)  # Predict using SVM
    return prediction  # Return predicted class

# Main execution
if __name__ == "__main__":
    uploaded_image_path = upload_image()  # Upload an image
    if uploaded_image_path:
        diagnosis = diagnose_allergy(uploaded_image_path)  # Diagnose the uploaded image

        # Map prediction to labels (update with your actual class names)
        class_names = train_generator.class_indices  # Get class indices
        inv_class_names = {v: k for k, v in class_names.items()}  # Inverse mapping
        result = inv_class_names[diagnosis[0]]  # Get the predicted class name

        # Display the result
        print(f"Diagnosis Result: {result}")

        # Display the uploaded image
        img = image.load_img(uploaded_image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Diagnosis Result: {result}")
        plt.show()
