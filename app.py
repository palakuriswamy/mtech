import os
import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ----------------------------
# Environment Settings
# ----------------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Load Tokenizer
# ----------------------------
print("Loading tokenizer...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded!")

# ----------------------------
# Load H5 Model (FIX)
# ----------------------------
print("Loading H5 model...")
model = load_model("fake_job_lstm_model.h5")
print("Model loaded successfully!")

# ----------------------------
# Constants
# ----------------------------
MAX_SEQUENCE_LENGTH = 200

# ----------------------------
# Text Preprocessing
# ----------------------------
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    return padded

# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    combined_text = request.form.get("combined_text")

    if not combined_text:
        return render_template(
            "index.html",
            prediction="Please enter the job description."
        )

    # Preprocess input
    input_data = preprocess_text(combined_text)

    # Predict using H5 model
    prediction = model.predict(input_data)[0][0]

    # Classification
    result = "Fraudulent" if prediction > 0.7 else "Legitimate"

    return render_template(
        "index.html",
        prediction=f"The job post is {result}"
    )

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
