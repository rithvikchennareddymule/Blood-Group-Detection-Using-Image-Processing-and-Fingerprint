from flask import Flask, request, render_template
import tensorflow as tf
import cv2
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model("blood_group_model.h5")  # Ensure this file is in the same folder
label_map = {0: "A+", 1: "A-", 2: "AB+", 3: "AB-", 4: "B+", 5: "B-", 6: "O+", 7: "O-"}  # Adjust based on your dataset


UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    # Save the uploaded file
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Preprocess the image
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128)) / 255.0  # Normalize pixel values
    img = img.reshape(1, 128, 128, 1)

    # Predict blood group
    predictions = model.predict(img)
    predicted_label = label_map[np.argmax(predictions)]

    return render_template("index.html", prediction=predicted_label, image_path=file_path)

if __name__ == "__main__":
    app.run(debug=True)
