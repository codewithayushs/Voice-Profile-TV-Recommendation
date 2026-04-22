"""
test_model.py  —  Test both gender and age models on an audio file
Usage:
    python test_model.py mrunali.mp3
    python test_model.py tiwari_voice.mp3
"""

import os
import sys
import numpy as np
import joblib
import librosa
import tensorflow as tf

# =============================
# CONFIG — paths relative to script location
# =============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GENDER_MODEL  = os.path.join(SCRIPT_DIR, "gender_model.keras")
GENDER_SCALER = os.path.join(SCRIPT_DIR, "gender_scaler.pkl")
AGE_MODEL     = os.path.join(SCRIPT_DIR, "age_model.keras")
AGE_SCALER    = os.path.join(SCRIPT_DIR, "age_scaler.pkl")

N_MFCC   = 40
SR       = 16000
DURATION = 3
HOP_LEN  = 512


def load_models():
    models = {}

    if os.path.exists(GENDER_MODEL) and os.path.exists(GENDER_SCALER):
        models['gender'] = {
            'model':  tf.keras.models.load_model(GENDER_MODEL),
            'data':   joblib.load(GENDER_SCALER)
        }
        le = models['gender']['data']['le']
        print(f"Gender model loaded  — classes: {le.classes_}")
    else:
        print("Gender model not found. Run: python voice_profile_training.py")

    if os.path.exists(AGE_MODEL) and os.path.exists(AGE_SCALER):
        models['age'] = {
            'model':  tf.keras.models.load_model(AGE_MODEL),
            'data':   joblib.load(AGE_SCALER)
        }
        le = models['age']['data']['le']
        print(f"Age model loaded     — classes: {le.classes_}")
    else:
        print("Age model not found. Run: python voice_profile_training.py")

    return models


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
        print(f"Feature extraction failed: {e}")
        return None


def predict_single(file_path, task_name, model, le, scaler):
    feats = extract_features(file_path)
    if feats is None:
        return None

    # Normalize
    for ch in range(feats.shape[-1]):
        ch_data = feats[:, :, ch].reshape(-1, 1)
        feats[:, :, ch] = scaler[ch].transform(ch_data).reshape(
            feats.shape[0], feats.shape[1]
        )

    pred      = model.predict(feats[np.newaxis, ...], verbose=0)[0]
    class_idx = np.argmax(pred)

    return {
        'label':      le.classes_[class_idx],
        'confidence': pred[class_idx] * 100,
        'all_probs':  dict(zip(le.classes_, [f"{p*100:.1f}%" for p in pred]))
    }


def predict_file(file_path, models):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"\nFile: {file_path}")
    print("-" * 40)

    results = {}

    if 'gender' in models:
        m    = models['gender']['model']
        data = models['gender']['data']
        r    = predict_single(file_path, 'gender', m, data['le'], data['scaler'])
        if r:
            results['gender'] = r
            print(f"Gender  : {r['label']:10s}  ({r['confidence']:.1f}% confidence)")
            print(f"  All   : {r['all_probs']}")

    if 'age' in models:
        m    = models['age']['model']
        data = models['age']['data']
        r    = predict_single(file_path, 'age', m, data['le'], data['scaler'])
        if r:
            results['age'] = r
            print(f"Age     : {r['label']:15s}  ({r['confidence']:.1f}% confidence)")
            print(f"  All   : {r['all_probs']}")

    if 'gender' in results and 'age' in results:
        g = results['gender']['label']
        a = results['age']['label']
        gc = results['gender']['confidence']
        ac = results['age']['confidence']
        print(f"\nFull profile: {g}, {a}  (avg confidence: {(gc+ac)/2:.1f}%)")


if __name__ == "__main__":
    models = load_models()

    if not models:
        print("No models loaded. Train first.")
        sys.exit(1)

    if len(sys.argv) > 1:
        predict_file(sys.argv[1], models)
    else:
        for f in ['mrunali.mp3', 'tiwari_voice.mp3']:
            fpath = os.path.join(SCRIPT_DIR, f)
            if os.path.exists(fpath):
                predict_file(fpath, models)
            else:
                print(f"Sample file not found: {f}")
 