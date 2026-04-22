import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA_DIR = "data/clips/cv-valid-dev"
METADATA_PATH = "data/metadata/cv-valid-dev.csv"
MODEL_DIR = "."
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_model.keras")
GENDER_SCALER = os.path.join(MODEL_DIR, "gender_scaler.pkl")
AGE_MODEL = os.path.join(MODEL_DIR, "age_model.keras")
AGE_SCALER = os.path.join(MODEL_DIR, "age_scaler.pkl")

# Feature extractor
import librosa

def load_models():
    """Load models and preprocessors"""
    print("Loading models...")
    gender_model = tf.keras.models.load_model(GENDER_MODEL)
    gender_data = joblib.load(GENDER_SCALER)
    age_model = tf.keras.models.load_model(AGE_MODEL)
    age_data = joblib.load(AGE_SCALER)
    
    print("Models loaded successfully")
    return {
        'gender': {'model': gender_model, 'scaler': gender_data['scaler'], 'le': gender_data['le']},
        'age': {'model': age_model, 'scaler': age_data['scaler'], 'le': age_data['le']}
    }

def extract_mfcc_features(file_path, sr=16000, duration=3.0, hop_length=512):
    """Extract MFCC+Delta+Delta2 matching model input (40, ~94, 3)"""
    y, _ = librosa.load(file_path, sr=sr, duration=duration + 0.5)
    y, _ = librosa.effects.trim(y, top_db=20)
    tlen = sr * duration
    pad_len = int(max(0, tlen - len(y)))
    y = np.pad(y, (0, pad_len))[:int(tlen)]
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.stack([mfcc, delta, delta2], axis=-1).astype(np.float32)

def evaluate_model(df, models, model_key):
    """Evaluate single model on validation set"""
    print(f"\nEvaluating {model_key} model...")
    
    y_true = []
    y_pred = []
    y_probs = []
    
    successful = 0
    total = len(df)
    
    for idx, row in df.iterrows():
        file_path = os.path.join(DATA_DIR, row['filename'])
        
        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            continue
            
        features = extract_mfcc_features(file_path)
        feats_scaled = preprocess_features(features, models[model_key]['scaler'])[np.newaxis]
        
        prob = models[model_key]['model'].predict(feats_scaled, verbose=0)[0]
        pred_idx = np.argmax(prob)
        pred_label = models[model_key]['le'].classes_[pred_idx]
        
        true_label = row[model_key]
        if pd.isna(true_label):
            continue
            
        y_true.append(true_label)
        y_pred.append(pred_label)
        y_probs.append(prob.max())
        
        successful += 1
        if successful % 50 == 0:
            print(f"Processed {successful}/{total}")
    
    print(f"Successfully processed {successful}/{total} files")
    
    if len(y_true) == 0:
        print("No valid predictions!")
        return None, None, None
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return y_true, y_pred, np.array(y_probs)

def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load data
    df = pd.read_csv(METADATA_PATH)
    print(f"Loaded {len(df)} validation samples")
    print("Columns:", df.columns.tolist())
    print("\nAge distribution:\n", df['age'].value_counts())
    print("\nGender distribution:\n", df['gender'].value_counts())
    
    # Load models
    models = load_models()
    
    # Evaluate gender
    y_true_g, y_pred_g, probs_g = evaluate_model(df, models, 'gender')
    if y_true_g is not None:
        plot_confusion_matrix(y_true_g, y_pred_g, models['gender']['le'].classes_, 'Gender Confusion Matrix')
    
    # Evaluate age
    y_true_a, y_pred_a, probs_a = evaluate_model(df, models, 'age')
    if y_true_a is not None:
        plot_confusion_matrix(y_true_a, y_pred_a, models['age']['le'].classes_, 'Age Confusion Matrix')
    
    print("\nEvaluation complete! Check generated plots.")
    print("Where models perform well:")
    print("- High diagonal values in confusion matrices")
    print("- Check classification reports for per-class performance")

if __name__ == "__main__":
    main()

