import numpy as np
import librosa
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import torchaudio

class VoiceAugmentor:
    """
    Real-time augmentation for voice profile training
    Handles class imbalance + data scarcity
    """
    
    def __init__(self, sample_rate=16000, augment_prob=0.5):
        self.sample_rate = sample_rate
        self.augment_prob = augment_prob
        
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(p=0.5),
        ])
    
    def augment_audio(self, audio_path, target_length=4.0):
        """Apply augmentations to raw audio → return augmented waveform"""
        try:
            # Load raw audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=target_length)
            
            # Pad/truncate
            target_samples = int(target_length * self.sample_rate)
            if len(y) < target_samples:
                y = np.pad(y, (0, target_samples - len(y)), 'constant')
            else:
                y = y[:target_samples]
            
            # Convert to torch tensor for audiomentations
            waveform = torch.from_numpy(y).float().unsqueeze(0)
            
            # Apply augmentations
            if np.random.rand() < self.augment_prob:
                augmented = self.augment(samples=waveform, sample_rate=self.sample_rate)
                y = augmented.squeeze().numpy()
            
            return y
            
        except Exception as e:
            print(f"Augmentation failed: {e}")
            return None

# Global augmenter instance
augmentor = VoiceAugmentor()

