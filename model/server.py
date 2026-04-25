from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import pickle
import os

from tensorflow.keras.models import load_model

# IMPORTANT (Render/HF crash fix)
os.environ["NUMBA_DISABLE_JIT"] = "1"

app = Flask(__name__)
CORS(app)

# Load models
gender_model = load_model("gender_model.keras",compile=False)
age_model = load_model("age_model.keras",compile=False)

# Load scalers
with open("gender_scaler.pkl", "rb") as f:
    gender_scaler = pickle.load(f)

with open("age_scaler.pkl", "rb") as f:
    age_scaler = pickle.load(f)


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/predict", methods=["POST"])
def predict():
    print("PREDICT HIT", flush=True)

    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    file = request.files["audio"]
    file_path = "temp.wav"
    file.save(file_path)

    # Load audio
    y, sr = librosa.load(file_path, sr=16000)

    # MFCC (same as training)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0).reshape(1, -1)

    # Scale
    gender_input = gender_scaler.transform(mfcc)
    age_input = age_scaler.transform(mfcc)

    # Predict
    gender_pred = gender_model.predict(gender_input)
    age_pred = age_model.predict(age_input)

    gender = "male" if gender_pred[0][0] > 0.5 else "female"

    age_classes = ["adult", "middle_aged", "senior"]
    age = age_classes[np.argmax(age_pred)]

    return jsonify({
        "gender": gender,
        "age_group": age,
        "gender_conf": float(np.max(gender_pred)),
        "age_conf": float(np.max(age_pred))
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)