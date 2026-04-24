"""
voice_profile_training.py  —  Speaker Attribute Classification v7.0
Trains TWO models:
    1. Gender model  -> predicts male / female
    2. Age model     ->  adult / mature_adult / senior

Both models are saved separately and used together in app.py and test_model.py
to give a combined prediction like: "male,adult"
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Dropout, LSTM, Reshape
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# CONFIGURATION
# =============================
BASE_PATH  = r"C:\Users\Ayush\OneDrive\Desktop\final dl project\data"
CSV_PATH   = os.path.join(BASE_PATH, "metadata", "cv-valid-train.csv")
AUDIO_PATH = os.path.join(BASE_PATH, "clips", "cv-valid-train")

MAX_SAMPLES_PER_CLASS = 5000
N_MFCC   = 40
SR       = 16000
DURATION = 3
HOP_LEN  = 512

# Replace your AGE_MAP with this simpler version
AGE_MAP = {
    'teens':     'adult',        # under ~20
    'twenties':  'adult',        # 20-29
    'thirties':  'middle_aged',  # 30-39
    'fourties':  'middle_aged',  # 40-49
    'fifties':   'middle_aged',  # 50-59
    'sixties':   'senior',       # 60+
    'seventies': 'senior',
    'eighties':  'senior'
}
print("=" * 55)
print("  Voice Attribute Classification — Gender + Age")
print("=" * 55)
print(f"CSV:   {CSV_PATH}")
print(f"Audio: {AUDIO_PATH}\n")

# =============================
# FEATURE EXTRACTION FUNCTION
# Shared by both gender and age training
# Returns shape (N_MFCC, T, 3):
#   channel 0 = MFCC
#   channel 1 = delta MFCC
#   channel 2 = delta-delta MFCC
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


# =============================
# MODEL BUILDER
# Same architecture for both tasks — only output size differs
# =============================
def build_model(input_shape, n_classes):
    inp = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, -1))(x)
    x = LSTM(64)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(n_classes, activation='softmax')(x)

    return Model(inp, out)


# =============================
# GENERIC TRAINING FUNCTION
# Used for both gender and age — avoids code duplication
# =============================
def run_training(task_name, df_task, label_col,
                 model_path, scaler_path, plot_prefix):

    print(f"\n{'='*55}")
    print(f"  Training: {task_name.upper()}")
    print(f"{'='*55}")
    print("Class distribution:")
    print(df_task[label_col].value_counts())

    # Balance classes
    frames = []
    for lbl, group in df_task.groupby(label_col):
        n = min(len(group), MAX_SAMPLES_PER_CLASS)
        frames.append(group.sample(n=n, random_state=42))
    df_balanced = pd.concat(frames).reset_index(drop=True)

    print(f"\nBalanced: {len(df_balanced)} samples")
    print(df_balanced[label_col].value_counts())

    # Extract features
    print(f"\nExtracting features...")
    X, y_str = [], []
    failed = 0

    for _, row in tqdm(df_balanced.iterrows(), total=len(df_balanced)):
        path = os.path.join(AUDIO_PATH, row['clean_filename'])
        if not os.path.exists(path):
            failed += 1
            continue
        feats = extract_features_3ch(path)
        if feats is not None:
            X.append(feats)
            y_str.append(row[label_col])
        else:
            failed += 1

    print(f"Extracted: {len(X)} success  |  {failed} failed")

    if len(X) < 20:
        print(f"Too few samples for {task_name}. Skipping.")
        return None

    X     = np.array(X)
    y_str = np.array(y_str)
    print(f"Feature shape: {X.shape}")

    # Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(y_str)
    n_classes = len(le.classes_)
    print(f"Classes ({n_classes}): {le.classes_}")

    # Normalize per channel
    scaler_dict = {}
    for ch in range(X.shape[-1]):
        ch_data = X[:, :, :, ch].reshape(-1, 1)
        scaler_dict[ch] = StandardScaler().fit(ch_data)
        X[:, :, :, ch] = scaler_dict[ch].transform(ch_data).reshape(
            X.shape[0], X.shape[1], X.shape[2]
        )

    joblib.dump({'le': le, 'scaler': scaler_dict}, scaler_path)
    print(f"Saved: {scaler_path}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # Build and compile
    model = build_model(X_train.shape[1:], n_classes)
    model.summary()

    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weight_dict)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-5, verbose=1),
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFINAL ACCURACY ({task_name}): {test_acc:.2%}")

    y_pred = model.predict(X_test).argmax(axis=1)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Training plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'],     label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title(f'{task_name} Accuracy')
    axes[0].legend()
    axes[1].plot(history.history['loss'],     label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_title(f'{task_name} Loss')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f'{plot_prefix}_training.png', dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(max(6, n_classes + 2), max(5, n_classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'{task_name} Confusion Matrix (Acc: {test_acc:.2%})')
    plt.tight_layout()
    plt.savefig(f'{plot_prefix}_confusion.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save model
    model.save(model_path)
    print(f"Saved: {model_path}")
    print(f"Saved: {plot_prefix}_training.png")
    print(f"Saved: {plot_prefix}_confusion.png")

    return test_acc


# =============================
# LOAD CSV ONCE — shared by both tasks
# =============================
df = pd.read_csv(CSV_PATH)
df['clean_filename'] = df['filename'].str.replace('cv-valid-train/', '', regex=False)

# =============================
# TASK 1: GENDER CLASSIFICATION
# Classes: female, male
# =============================
df_gender = df[df['gender'].notna() & df['gender'].isin(['male', 'female'])].copy()
df_gender['label_gender'] = df_gender['gender']

gender_acc = run_training(
    task_name   = "Gender",
    df_task     = df_gender,
    label_col   = "label_gender",
    model_path  = "gender_model.keras",
    scaler_path = "gender_scaler.pkl",
    plot_prefix = "gender"
)

# =============================
# TASK 2: AGE CLASSIFICATION
# Classes: teenager, young_adult, adult, mature_adult, senior
# =============================
df_age = df[df['age'].notna()].copy()
df_age['label_age'] = df_age['age'].map(AGE_MAP)
df_age = df_age[df_age['label_age'].notna()]

age_acc = run_training(
    task_name   = "Age",
    df_task     = df_age,
    label_col   = "label_age",
    model_path  = "age_model.keras",
    scaler_path = "age_scaler.pkl",
    plot_prefix = "age"
)

# =============================
# SUMMARY
# =============================
print("\n" + "=" * 55)
print("  TRAINING COMPLETE — BOTH MODELS")
print("=" * 55)
if gender_acc:
    print(f"  Gender model accuracy : {gender_acc:.2%}  -> gender_model.keras")
if age_acc:
    print(f"  Age model accuracy    : {age_acc:.2%}  -> age_model.keras")
print()

