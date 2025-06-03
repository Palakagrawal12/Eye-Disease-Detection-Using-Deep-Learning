from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "xception.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
class_labels = ["Normal", "Cataract", "Diabetic Retinopathy", "Glaucoma"]

# Ensure upload directory exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Home Page - Upload Form
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if an image was uploaded
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded.")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", error="No selected file.")

        # Save uploaded image
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Preprocess image
        img = image.load_img(file_path, target_size=(299, 299))
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100  # Confidence score

        return render_template("index.html", prediction=predicted_class, confidence=f"{confidence:.2f}", img_path=file_path)

    return render_template("index.html")


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
