"""
show_results.py

Displays dataset shapes and evaluates saved Gender & Age models
on the official training (sample), validation (cv-valid-dev), and test (cv-valid-test) splits.

Outputs:
    - Dataset shapes (train / dev / test)
    - Training accuracy    (evaluated on a stratified sample from cv-valid-train)
    - Validation accuracy  (evaluated on cv-valid-dev)
    - Test accuracy        (evaluated on cv-valid-test)
    - Single bar chart comparing all three accuracies
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# =============================
# CONFIGURATION
# =============================
BASE_PATH   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_PATH, "data")
METADATA_PATH = os.path.join(DATA_PATH, "metadata")
CLIPS_PATH    = os.path.join(DATA_PATH, "clips")

N_MFCC   = 40
SR       = 16000
DURATION = 3
HOP_LEN  = 512

# How many training samples per class to evaluate (to keep runtime reasonable)
MAX_TRAIN_PER_CLASS = 2000

AGE_MAP = {
    'teens':     'adult',
    'twenties':  'adult',
    'thirties':  'middle_aged',
    'fourties':  'middle_aged',
    'fifties':   'middle_aged',
    'sixties':   'senior',
    'seventies': 'senior',
    'eighties':  'senior'
}

# =============================
# FEATURE EXTRACTION (same as training)
# =============================
def extract_features_3ch(file_path):
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
    except Exception:
        return None


def apply_scaler(X, scaler_dict):
    """Apply per-channel StandardScaler (same logic as training)."""
    X_scaled = X.copy()
    for ch in range(X.shape[-1]):
        ch_data = X_scaled[:, :, :, ch].reshape(-1, 1)
        X_scaled[:, :, :, ch] = scaler_dict[ch].transform(ch_data).reshape(
            X.shape[0], X.shape[1], X.shape[2]
        )
    return X_scaled


def load_and_prepare(csv_path, audio_dir, label_col, allowed_values=None, age_map=None):
    """
    Load a CSV, filter rows with valid labels & existing audio files,
    and return (DataFrame, audio_paths, labels).
    """
    df = pd.read_csv(csv_path)
    df['clean_filename'] = df['filename'].apply(lambda x: os.path.basename(x))

    # Filter valid labels
    mask = df[label_col].notna()
    if allowed_values is not None:
        mask &= df[label_col].isin(allowed_values)
    df = df[mask].copy()

    if age_map is not None:
        df[label_col] = df[label_col].map(age_map)
        df = df[df[label_col].notna()]

    audio_paths = []
    labels = []
    for _, row in df.iterrows():
        path = os.path.join(audio_dir, row['clean_filename'])
        if os.path.exists(path):
            audio_paths.append(path)
            labels.append(row[label_col])

    return df, audio_paths, np.array(labels)


def evaluate_on_split(model, scaler_dict, le, audio_paths, y_labels, task_name, split_name, tqdm_desc):
    """
    Extract features, normalize, evaluate, and return accuracy.
    Returns accuracy float or None.
    """
    if len(audio_paths) == 0:
        print(f"   → {task_name} {split_name} Accuracy : N/A (no samples)")
        return None

    X = []
    valid_idx = []
    for i, path in enumerate(tqdm(audio_paths, desc=tqdm_desc, ncols=70)):
        feats = extract_features_3ch(path)
        if feats is not None:
            X.append(feats)
            valid_idx.append(i)

    if len(X) == 0:
        print(f"   → {task_name} {split_name} Accuracy : N/A (feature extraction failed)")
        return None

    X = np.array(X)
    y_enc = le.transform(y_labels[valid_idx])
    X = apply_scaler(X, scaler_dict)
    _, acc = model.evaluate(X, y_enc, verbose=0)
    print(f"   → {task_name} {split_name} Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    return acc


def sample_train_paths(audio_paths, y_labels, max_per_class=MAX_TRAIN_PER_CLASS):
    """Stratified random sample of training paths to keep runtime sane."""
    df_tmp = pd.DataFrame({'path': audio_paths, 'label': y_labels})
    sampled = []
    sampled_labels = []
    for lbl, group in df_tmp.groupby('label'):
        n = min(len(group), max_per_class)
        chosen = group.sample(n=n, random_state=42)
        sampled.extend(chosen['path'].tolist())
        sampled_labels.extend(chosen['label'].tolist())
    return sampled, np.array(sampled_labels)


def plot_accuracy_comparison(results, save_path="accuracy_comparison.png"):
    """
    results = {
        'Gender': {'train': 0.97, 'val': 0.95, 'test': 0.96},
        'Age':    {'train': 0.72, 'val': 0.60, 'test': 0.62},
    }
    """
    tasks = list(results.keys())
    metrics = ['train', 'val', 'test']
    colors = {'train': '#2ecc71', 'val': '#f39c12', 'test': '#3498db'}
    labels = {'train': 'Training', 'val': 'Validation', 'test': 'Test'}

    x = np.arange(len(tasks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(metrics):
        vals = [results[t][m] if results[t][m] is not None else 0 for t in tasks]
        bars = ax.bar(x + i*width, vals, width, label=labels[m], color=colors[m], edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val is not None and val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val*100:.1f}%", ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n   📊 Saved accuracy comparison chart: {save_path}")


# =============================
# MAIN
# =============================
def main():
    print("=" * 60)
    print("  DATASET & MODEL RESULTS SUMMARY")
    print("=" * 60)

    # -------------------------------------------------
    # 1. Dataset Shapes
    # -------------------------------------------------
    csv_train = os.path.join(METADATA_PATH, "cv-valid-train.csv")
    csv_dev   = os.path.join(METADATA_PATH, "cv-valid-dev.csv")
    csv_test  = os.path.join(METADATA_PATH, "cv-valid-test.csv")

    df_train = pd.read_csv(csv_train)
    df_dev   = pd.read_csv(csv_dev)
    df_test  = pd.read_csv(csv_test)

    print("\n1. DATASET SHAPES")
    print("-" * 40)
    print(f"   Train CSV : {df_train.shape[0]:,} rows × {df_train.shape[1]} cols")
    print(f"   Dev CSV   : {df_dev.shape[0]:,} rows × {df_dev.shape[1]} cols")
    print(f"   Test CSV  : {df_test.shape[0]:,} rows × {df_test.shape[1]} cols")

    train_gender = df_train['gender'].isin(['male', 'female']).sum()
    train_age    = df_train['age'].notna().sum()
    dev_gender   = df_dev['gender'].isin(['male', 'female']).sum()
    dev_age      = df_dev['age'].notna().sum()
    test_gender  = df_test['gender'].isin(['male', 'female']).sum()
    test_age     = df_test['age'].notna().sum()

    print(f"\n   Labeled samples:")
    print(f"   {'Split':<10} {'Gender':>10} {'Age':>10}")
    print(f"   {'Train':<10} {train_gender:>10,} {train_age:>10,}")
    print(f"   {'Dev':<10} {dev_gender:>10,} {dev_age:>10,}")
    print(f"   {'Test':<10} {test_gender:>10,} {test_age:>10,}")

    # -------------------------------------------------
    # 2. Load Models & Scalers
    # -------------------------------------------------
    print("\n2. LOAD MODELS & SCALERS")
    print("-" * 40)

    gender_model_path = os.path.join(BASE_PATH, "gender_model.keras")
    age_model_path    = os.path.join(BASE_PATH, "age_model.keras")
    gender_scaler_path = os.path.join(BASE_PATH, "gender_scaler.pkl")
    age_scaler_path    = os.path.join(BASE_PATH, "age_scaler.pkl")

    if not os.path.exists(gender_model_path):
        print(f"   [!] Gender model not found: {gender_model_path}")
        return
    if not os.path.exists(age_model_path):
        print(f"   [!] Age model not found: {age_model_path}")
        return

    gender_model = tf.keras.models.load_model(gender_model_path)
    age_model    = tf.keras.models.load_model(age_model_path)
    print("   ✓ Gender model loaded")
    print("   ✓ Age model loaded")

    gender_bundle = joblib.load(gender_scaler_path)
    age_bundle    = joblib.load(age_scaler_path)
    gender_le = gender_bundle['le']
    gender_scaler = gender_bundle['scaler']
    age_le    = age_bundle['le']
    age_scaler = age_bundle['scaler']
    print("   ✓ Scalers & encoders loaded")

    # Prepare result dicts
    results = {
        'Gender': {'train': None, 'val': None, 'test': None},
        'Age':    {'train': None, 'val': None, 'test': None},
    }

    # -------------------------------------------------
    # 3. Training Accuracy (sampled from cv-valid-train)
    # -------------------------------------------------
    print("\n3. TRAINING ACCURACY  (stratified sample from cv-valid-train)")
    print("-" * 40)

    train_audio_dir = os.path.join(CLIPS_PATH, "cv-valid-train")

    # Gender train sample
    _, paths_gtrain, y_gtrain = load_and_prepare(
        csv_train, train_audio_dir, 'gender', allowed_values=['male', 'female']
    )
    paths_gtrain_samp, y_gtrain_samp = sample_train_paths(paths_gtrain, y_gtrain, MAX_TRAIN_PER_CLASS)
    print(f"   Gender train samples (sampled): {len(paths_gtrain_samp)}")
    results['Gender']['train'] = evaluate_on_split(
        gender_model, gender_scaler, gender_le,
        paths_gtrain_samp, y_gtrain_samp,
        "Gender", "Training", "   Gender train"
    )

    # Age train sample
    _, paths_atrain, y_atrain = load_and_prepare(
        csv_train, train_audio_dir, 'age', age_map=AGE_MAP
    )
    paths_atrain_samp, y_atrain_samp = sample_train_paths(paths_atrain, y_atrain, MAX_TRAIN_PER_CLASS)
    print(f"   Age train samples (sampled): {len(paths_atrain_samp)}")
    results['Age']['train'] = evaluate_on_split(
        age_model, age_scaler, age_le,
        paths_atrain_samp, y_atrain_samp,
        "Age", "Training", "   Age train"
    )

    # -------------------------------------------------
    # 4. Validation Accuracy (cv-valid-dev)
    # -------------------------------------------------
    print("\n4. VALIDATION ACCURACY  (cv-valid-dev)")
    print("-" * 40)

    dev_audio_dir = os.path.join(CLIPS_PATH, "cv-valid-dev", "cv-valid-dev")

    _, paths_gdev, y_gdev = load_and_prepare(
        csv_dev, dev_audio_dir, 'gender', allowed_values=['male', 'female']
    )
    print(f"   Gender dev samples with audio: {len(paths_gdev)}")
    results['Gender']['val'] = evaluate_on_split(
        gender_model, gender_scaler, gender_le,
        paths_gdev, y_gdev,
        "Gender", "Validation", "   Gender dev"
    )

    _, paths_adev, y_adev = load_and_prepare(
        csv_dev, dev_audio_dir, 'age', age_map=AGE_MAP
    )
    print(f"   Age dev samples with audio: {len(paths_adev)}")
    results['Age']['val'] = evaluate_on_split(
        age_model, age_scaler, age_le,
        paths_adev, y_adev,
        "Age", "Validation", "   Age dev"
    )

    # -------------------------------------------------
    # 5. Test Accuracy (cv-valid-test)
    # -------------------------------------------------
    print("\n5. TEST ACCURACY  (cv-valid-test)")
    print("-" * 40)

    test_audio_dir = os.path.join(CLIPS_PATH, "cv-valid-test", "cv-valid-test")

    _, paths_gtest, y_gtest = load_and_prepare(
        csv_test, test_audio_dir, 'gender', allowed_values=['male', 'female']
    )
    print(f"   Gender test samples with audio: {len(paths_gtest)}")
    results['Gender']['test'] = evaluate_on_split(
        gender_model, gender_scaler, gender_le,
        paths_gtest, y_gtest,
        "Gender", "Test", "   Gender test"
    )

    _, paths_atest, y_atest = load_and_prepare(
        csv_test, test_audio_dir, 'age', age_map=AGE_MAP
    )
    print(f"   Age test samples with audio: {len(paths_atest)}")
    results['Age']['test'] = evaluate_on_split(
        age_model, age_scaler, age_le,
        paths_atest, y_atest,
        "Age", "Test", "   Age test"
    )

    # -------------------------------------------------
    # 6. Plot comparison
    # -------------------------------------------------
    print("\n6. PLOTTING")
    print("-" * 40)
    plot_accuracy_comparison(results, save_path="accuracy_comparison.png")

    # -------------------------------------------------
    # 7. Summary
    # -------------------------------------------------
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Dataset shapes:")
    print(f"    Train : {df_train.shape}")
    print(f"    Dev   : {df_dev.shape}")
    print(f"    Test  : {df_test.shape}")
    print(f"  Feature shape : (40, ~94, 3)  [MFCC + Δ + ΔΔ]")
    print(f"  Models        : gender_model.keras, age_model.keras")
    for task in ['Gender', 'Age']:
        train_str = f"{results[task]['train']*100:.2f}%" if results[task]['train'] is not None else "N/A"
        val_str   = f"{results[task]['val']*100:.2f}%"   if results[task]['val']   is not None else "N/A"
        test_str  = f"{results[task]['test']*100:.2f}%"  if results[task]['test']  is not None else "N/A"
        print(f"  {task:<10} Train: {train_str} | Val: {val_str} | Test: {test_str}")
    print("=" * 60)


if __name__ == "__main__":
    main()

