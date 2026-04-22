import librosa
import numpy as np

def extract_features(file_path):
    try:
        from utils.full_feature_extractor import extract_features as full_extract
        return full_extract(file_path)
    except ImportError:
        # Fallback to original if full not available
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack((mfcc, delta, delta2))
        return np.mean(features.T, axis=0)
