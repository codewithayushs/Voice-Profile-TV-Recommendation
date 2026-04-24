import os
import tempfile
import subprocess
import numpy as np
import librosa
import joblib
import tensorflow as tf
import imageio_ffmpeg
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# =============================
# CONFIG
# =============================
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
GENDER_MODEL  = os.path.join(SCRIPT_DIR, "gender_model.keras")
GENDER_SCALER = os.path.join(SCRIPT_DIR, "gender_scaler.pkl")
AGE_MODEL     = os.path.join(SCRIPT_DIR, "age_model.keras")
AGE_SCALER    = os.path.join(SCRIPT_DIR, "age_scaler.pkl")

N_MFCC   = 40
SR       = 16000
DURATION = 3
HOP_LEN  = 512

# =============================
# LOAD MODELS
# =============================
print("Loading models...")
gender_model = tf.keras.models.load_model(GENDER_MODEL)
gender_data  = joblib.load(GENDER_SCALER)
age_model    = tf.keras.models.load_model(AGE_MODEL)
age_data     = joblib.load(AGE_SCALER)
print("Gender classes:", gender_data['le'].classes_)
print("Age classes   :", age_data['le'].classes_)
print("Models loaded!\n")

# =============================
# AUDIO CONVERT
# =============================
def convert_to_wav(input_path, output_path):
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run(
            [ffmpeg_exe, "-y", "-i", input_path, "-ar", str(SR), "-ac", "1", output_path],
            capture_output=True
        )
        return result.returncode == 0
    except Exception as e:
        print("Conversion error:", e)
        return False

# =============================
# FEATURE EXTRACTION
# =============================
def extract_features(file_path):
    y, _ = librosa.load(file_path, sr=SR, duration=DURATION + 0.5)
    y, _ = librosa.effects.trim(y, top_db=20)

    target_len = SR * DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mfcc   = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LEN)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    return np.stack([mfcc, delta, delta2], axis=-1).astype(np.float32)

# =============================
# PREDICT
# =============================
def predict(feats, model, le, scaler):
    f = feats.copy()

    for ch in range(f.shape[-1]):
        ch_data = f[:, :, ch].reshape(-1, 1)
        f[:, :, ch] = scaler[ch].transform(ch_data).reshape(f.shape[0], f.shape[1])

    pred = model.predict(f[np.newaxis, ...], verbose=0)[0]

    class_idx = np.argmax(pred)
    confidence = float(pred[class_idx])
    label = le.classes_[class_idx]

    all_probs = {le.classes_[i]: float(pred[i]) for i in range(len(le.classes_))}

    return label, confidence, all_probs

# =============================
# ROUTE
# =============================
@app.route("/predict", methods=["POST"])
def predict_route():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio_file.save(tmp.name)
        webm_path = tmp.name

    wav_path = webm_path.replace(".webm", ".wav")

    try:
        if not convert_to_wav(webm_path, wav_path):
            return jsonify({"error": "Conversion failed"}), 500

        feats = extract_features(wav_path)

        gender_label, gender_conf, gender_probs = predict(
            feats, gender_model, gender_data['le'], gender_data['scaler']
        )

        age_label, age_conf, age_probs = predict(
            feats, age_model, age_data['le'], age_data['scaler']
        )

        # ✅ FIXED MAPPING (ALIGNED WITH TRAINING)
        age_group_map = {
            "adult":         "adult",
            "middle_aged":   "middle_aged",
            "senior":        "senior",

            # backward compatibility
            "young":         "adult",
            "mature_adult":  "middle_aged",
        }

        age_group = age_group_map.get(age_label, age_label)

        return jsonify({
            "gender":       gender_label,
            "gender_conf":  gender_conf,
            "gender_probs": gender_probs,

            "age_label":    age_label,
            "age_group":    age_group,
            "age_conf":     age_conf,
            "age_probs":    age_probs,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        for path in [webm_path, wav_path]:
            if os.path.exists(path):
                os.remove(path)

# =============================
# HEALTH
# =============================
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Server running at http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
