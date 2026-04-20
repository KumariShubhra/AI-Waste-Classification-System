from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("waste_classification_model.h5")

# Define class labels
CLASS_LABELS = {0: "metal", 1: "organic", 2: "other", 3: "paper", 4: "plastic"}

# Preprocess image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to match model input
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    image = Image.open(file)
    processed_image = preprocess_image(image)

    # Predict
    predictions = model.predict(processed_image)
    print("Raw Model Output:", predictions)  # Optional: useful for debugging

    # Find the class with highest probability
    predicted_index = np.argmax(predictions[0])
    predicted_label = CLASS_LABELS[predicted_index]

    # Optionally print all class probabilities
    for idx, score in enumerate(predictions[0]):
        print(f"{CLASS_LABELS[idx]}: {score:.4f}")

    return render_template("index.html", prediction=predicted_label)

if __name__ == "__main__":
    app.run(debug=True)





