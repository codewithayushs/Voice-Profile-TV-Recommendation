import streamlit as st
import os
import io
import numpy as np
import librosa
import joblib
import tensorflow as tf
from streamlit_mic_recorder import mic_recorder

# =============================
# CONFIG (Matching your file)
# =============================
N_MFCC   = 40
SR       = 16000
DURATION = 3
HOP_LEN  = 512

# =============================
# MODEL LOADING (Cached)
# =============================
@st.cache_resource
def load_all_models():
    """Loads models once and keeps them in memory."""
    models = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths matching your record_and_predict.py
    paths = {
        'gender': (os.path.join(script_dir, "gender_model.keras"), os.path.join(script_dir, "gender_scaler.pkl")),
        'age':    (os.path.join(script_dir, "age_model.keras"), os.path.join(script_dir, "age_scaler.pkl"))
    }

    for key, (m_path, s_path) in paths.items():
        if os.path.exists(m_path) and os.path.exists(s_path):
            models[key] = {
                'model': tf.keras.models.load_model(m_path),
                'data':  joblib.load(s_path)
            }
    return models

# =============================
# FEATURE EXTRACTION
# =============================
def extract_features_web(audio_bytes):
    """Processes audio bytes directly from the browser."""
    try:
        # Load from memory instead of disk
        audio_file = io.BytesIO(audio_bytes)
        y, _ = librosa.load(audio_file, sr=SR, duration=DURATION + 0.5)
        
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
        st.error(f"Feature extraction error: {e}")
        return None

# =============================
# PREDICTION LOGIC
# =============================
def run_prediction(feats, model_dict):
    m = model_dict['model']
    data = model_dict['data']
    le = data['le']
    scaler = data['scaler']

    f = feats.copy()
    for ch in range(f.shape[-1]):
        ch_data = f[:, :, ch].reshape(-1, 1)
        f[:, :, ch] = scaler[ch].transform(ch_data).reshape(f.shape[0], f.shape[1])
    
    pred = m.predict(f[np.newaxis, ...], verbose=0)[0]
    class_idx = np.argmax(pred)
    
    results = {le.classes_[i]: float(pred[i] * 100) for i in range(len(le.classes_))}
    return le.classes_[class_idx], results

# =============================
# STREAMLIT UI
# =============================
st.set_page_config(page_title="Voice Profile AI", page_icon="🎙️")

st.title("🎙️ Voice Profile Predictor")
st.markdown("Predict **Gender** and **Age Group** from live audio.")

models = load_all_models()

if not models:
    st.error("Models not found! Ensure .keras and .pkl files are in the same folder.")
else:
    # Sidebar Info
    st.sidebar.header("Model Status")
    for key in ['gender', 'age']:
        status = "✅ Loaded" if key in models else "❌ Missing"
        st.sidebar.write(f"**{key.capitalize()}:** {status}")

    # Mic Recording Component
    st.write("### Step 1: Record your voice")
    audio = mic_recorder(
        start_prompt="Click to Start Recording",
        stop_prompt="Stop Recording",
        key='recorder'
    )

    if audio:
        st.audio(audio['bytes'])
        
        if st.button("Analyze Recording"):
            with st.spinner("Processing audio features..."):
                feats = extract_features_web(audio['bytes'])
            
            if feats is not None:
                col1, col2 = st.columns(2)

                # --- Gender Results ---
                if 'gender' in models:
                    label, probs = run_prediction(feats, models['gender'])
                    with col1:
                        st.metric("Predicted Gender", label.upper())
                        for cls, val in probs.items():
                            st.write(f"{cls}:")
                            st.progress(val / 100)
                
                # --- Age Results ---
                if 'age' in models:
                    label, probs = run_prediction(feats, models['age'])
                    with col2:
                        st.metric("Predicted Age", label.upper())
                        for cls, val in probs.items():
                            st.write(f"{cls}:")
                            st.progress(val / 100)
            else:
                st.warning("Could not process the audio. Please try speaking more clearly.")