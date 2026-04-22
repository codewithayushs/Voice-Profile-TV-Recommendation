"""
record_and_predict.py  —  Record your voice live and predict gender + age
Usage:
    python record_and_predict.py              # default 5 seconds
    python record_and_predict.py --duration 8 # record 8 seconds
    python record_and_predict.py --rounds 3   # record and predict 3 times
"""

import os
import sys
import time
import argparse
import numpy as np
import librosa
import joblib
import tensorflow as tf

# =============================
# CHECK DEPENDENCIES
# =============================
try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    print("Missing libraries. Run:")
    print("  pip install sounddevice soundfile")
    sys.exit(1)

# =============================
# CONFIG
# =============================
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
GENDER_MODEL  = os.path.join(SCRIPT_DIR, "gender_model.keras")
GENDER_SCALER = os.path.join(SCRIPT_DIR, "gender_scaler.pkl")
AGE_MODEL     = os.path.join(SCRIPT_DIR, "age_model.keras")
AGE_SCALER    = os.path.join(SCRIPT_DIR, "age_scaler.pkl")
TEMP_FILE     = os.path.join(SCRIPT_DIR, "_temp_recording.wav")

N_MFCC   = 40
SR       = 16000
DURATION = 3
HOP_LEN  = 512

# =============================
# LOAD MODELS
# =============================
def load_models():
    models = {}
    if os.path.exists(GENDER_MODEL) and os.path.exists(GENDER_SCALER):
        models['gender'] = {
            'model': tf.keras.models.load_model(GENDER_MODEL),
            'data':  joblib.load(GENDER_SCALER)
        }
        print(f"  Gender model loaded  classes: {models['gender']['data']['le'].classes_}")
    else:
        print("  Gender model not found — run voice_profile_training.py first")

    if os.path.exists(AGE_MODEL) and os.path.exists(AGE_SCALER):
        models['age'] = {
            'model': tf.keras.models.load_model(AGE_MODEL),
            'data':  joblib.load(AGE_SCALER)
        }
        print(f"  Age model loaded     classes: {models['age']['data']['le'].classes_}")
    else:
        print("  Age model not found — run voice_profile_training.py first")

    return models

# =============================
# FEATURE EXTRACTION
# =============================
def extract_features(file_path):
    try:
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
    except Exception as e:
        print(f"  Feature extraction error: {e}")
        return None

# =============================
# PREDICT
# =============================
def predict(feats, model, le, scaler):
    f = feats.copy()
    for ch in range(f.shape[-1]):
        ch_data = f[:, :, ch].reshape(-1, 1)
        f[:, :, ch] = scaler[ch].transform(ch_data).reshape(f.shape[0], f.shape[1])
    pred      = model.predict(f[np.newaxis, ...], verbose=0)[0]
    class_idx = np.argmax(pred)
    all_probs = {le.classes_[i]: pred[i] * 100 for i in range(len(le.classes_))}
    return le.classes_[class_idx], pred[class_idx] * 100, all_probs

# =============================
# CONFIDENCE BAR
# =============================
def conf_bar(pct, width=20):
    filled = int(pct / 100 * width)
    bar    = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {pct:.1f}%"

# =============================
# RECORD + PREDICT ONE ROUND
# =============================
def record_and_predict(models, duration, round_num=None):
    label = f"Round {round_num}" if round_num else "Recording"

    print(f"\n{'='*50}")
    if round_num:
        print(f"  {label}")
    print(f"{'='*50}")

    # Countdown
    for i in range(3, 0, -1):
        print(f"  Starting in {i}...", end="\r")
        time.sleep(1)

    print(f"  RECORDING — speak now ({duration}s)          ")
    print("  " + "-" * 30)

    # Record
    audio = sd.rec(
        int(duration * SR),
        samplerate=SR,
        channels=1,
        dtype='float32'
    )
    # Live progress bar while recording
    for i in range(duration):
        filled = int((i + 1) / duration * 20)
        bar    = "#" * filled + "-" * (20 - filled)
        print(f"  [{bar}] {i+1}/{duration}s", end="\r")
        time.sleep(1)
    sd.wait()
    print(f"  [####################] Done!              ")

    # Save to temp file
    sf.write(TEMP_FILE, audio, SR)

    # Extract features
    feats = extract_features(TEMP_FILE)
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)

    if feats is None:
        print("  Could not process audio. Try again.")
        return

    # Run predictions
    print()
    results = {}

    if 'gender' in models:
        m      = models['gender']['model']
        data   = models['gender']['data']
        label_pred, conf, probs = predict(feats, m, data['le'], data['scaler'])
        results['gender'] = (label_pred, conf)
        print(f"  Gender  : {label_pred.upper()}")
        for cls, pct in sorted(probs.items(), key=lambda x: -x[1]):
            marker = " <--" if cls == label_pred else ""
            print(f"    {cls:12s} {conf_bar(pct)}{marker}")

    print()

    if 'age' in models:
        m      = models['age']['model']
        data   = models['age']['data']
        label_pred, conf, probs = predict(feats, m, data['le'], data['scaler'])
        results['age'] = (label_pred, conf)
        print(f"  Age     : {label_pred.upper()}")
        for cls, pct in sorted(probs.items(), key=lambda x: -x[1]):
            marker = " <--" if cls == label_pred else ""
            print(f"    {cls:12s} {conf_bar(pct)}{marker}")

    # Combined result
    if 'gender' in results and 'age' in results:
        g, gc = results['gender']
        a, ac = results['age']
        avg   = (gc + ac) / 2
        print()
        print(f"  PROFILE : {g}, {a}")
        print(f"  Avg confidence: {avg:.1f}%")

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record voice and predict gender + age")
    parser.add_argument("--duration", type=int, default=5,
                        help="Recording duration in seconds (default: 5)")
    parser.add_argument("--rounds",   type=int, default=1,
                        help="Number of recording rounds (default: 1)")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("  Voice Profile Predictor — Live Recording")
    print("="*50)
    print("\nLoading models...")
    models = load_models()

    if not models:
        print("\nNo models found. Run: python voice_profile_training.py")
        sys.exit(1)

    print(f"\nSettings: {args.duration}s recording, {args.rounds} round(s)")
    print("Tip: speak clearly, avoid background noise")

    if args.rounds == 1:
        record_and_predict(models, args.duration)
    else:
        for i in range(1, args.rounds + 1):
            record_and_predict(models, args.duration, round_num=i)
            if i < args.rounds:
                input("\n  Press ENTER for next round...")

    print("\n" + "="*50)
    print("  Done!")
    print("="*50)
