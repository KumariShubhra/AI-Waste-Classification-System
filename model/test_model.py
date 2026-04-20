import tensorflow as tf
import numpy as np
import cv2

# Load the model
model = tf.keras.models.load_model("waste_classifier_model.h5")
print("Model loaded successfully!")

# Load and preprocess the test image
img_path = "test_image5.jpg"  # Make sure this file exists in the same directory
img = cv2.imread(img_path)   # Read the image
img = cv2.resize(img, (150, 150))  # Resize to 150x150 as required by the model
img = img.astype('float32') / 255.0  # Normalize pixel values
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Check image shape
print(f"Processed Image Shape: {img.shape}")  # Should print (1, 150, 150, 3)

# Make predictions
predictions = model.predict(img)
predicted_class = np.argmax(predictions)

# Print the result
print(f"Predicted Class: {predicted_class}")
