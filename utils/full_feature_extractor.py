import librosa
import numpy as np

def extract_features(file_path, target_length=4.0, sr=16000):
    """
    Robust LogMel 128x64 for CNN - FIXED for dataset
    Returns (128,64,1) or None
    """
    try:
        print(f"Loading: {file_path}")  # Debug
        y, _ = librosa.load(file_path, sr=sr, duration=target_length, mono=True)
        
        # Pad/truncate exactly 4s
        target_samples = int(target_length * sr)
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)), 'constant')
        else:
            y = y[:target_samples]
        
        print(f"Audio shape: {y.shape}, duration: {len(y)/sr:.2f}s")
        
        # Fixed Log-Mel for exactly 64 frames
        hop_length = int(sr * target_length / 64)  # 16000*4/64 = 1000
        n_fft = min(2048, hop_length * 4)
        
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, fmax=8000,
            n_fft=n_fft, hop_length=hop_length
        )
        log_mel = librosa.amplitude_to_db(mel_spec, ref=np.max)
        
        print(f"Mel shape: {log_mel.shape}")  # Should be (128,64)
        
        # Ensure exact shape
        if log_mel.shape[1] != 64:
            if log_mel.shape[1] < 64:
                pad_w = 64 - log_mel.shape[1]
                log_mel = np.pad(log_mel, ((0,0), (0, pad_w)), 'constant')
            else:
                log_mel = log_mel[:, :64]
        
        # CNN-ready: (128,64,1)
        features = log_mel[:, :, np.newaxis]
        print(f"✅ Features extracted: {features.shape}")
        return features
        
    except Exception as e:
        print(f"❌ Feature extraction FAILED: {e}")
        return None

