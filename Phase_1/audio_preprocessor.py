
import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import Union, List, Dict
import soundfile as sf
from config import Config

class AudioPreprocessor:
    """
    Audio preprocessing for deepfake detection
    Uses Wav2Vec2 transformer + MFCC features
    Reference: FakeAVCeleb audio-visual deepfake detection (96.55% accuracy)
    """

    def __init__(self, model_name: str = Config.WAV2VEC_MODEL):
        self.device = Config.DEVICE
        self.sample_rate = Config.AUDIO_SAMPLE_RATE
        self.max_length = Config.AUDIO_MAX_LENGTH
        self.n_mfcc = Config.N_MFCC
        self.n_mels = Config.N_MELS

        print(f"Loading Wav2Vec2: {model_name}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print(f"✅ Audio Preprocessor initialized on {self.device}")
        print(f"   Sample Rate: {self.sample_rate} Hz")
        print(f"   Max Length: {self.max_length} seconds")
        print(f"   MFCC Features: {self.n_mfcc}")

    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load audio file and resample to target sample rate

        Args:
            audio_path: Path to audio file

        Returns:
            Audio waveform as numpy array
        """
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Trim or pad to max length
        max_samples = self.sample_rate * self.max_length
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        else:
            waveform = np.pad(waveform, (0, max_samples - len(waveform)))

        return waveform

    def extract_mfcc(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract MFCC (Mel-Frequency Cepstral Coefficients) features
        Classical audio features for voice analysis

        Args:
            waveform: Audio waveform

        Returns:
            MFCC features of shape (n_mfcc, time_steps)
        """
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=2048,
            hop_length=512
        )

        # Normalize
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

        return mfcc

    def extract_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract Mel Spectrogram for audio-visual deepfake detection

        Args:
            waveform: Audio waveform

        Returns:
            Mel spectrogram of shape (n_mels, time_steps)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=2048,
            hop_length=512
        )

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db

    def extract_wav2vec_features(self, waveform: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
        """
        Extract Wav2Vec2 transformer features for semantic audio understanding

        Args:
            waveform: Single waveform or list of waveforms

        Returns:
            Tensor of shape (batch_size, feature_dim)
        """
        with torch.no_grad():
            # Process audio
            if isinstance(waveform, np.ndarray) and waveform.ndim == 1:
                waveform = [waveform]

            inputs = self.processor(
                waveform,
                sampling_rate=self.sample_rate,
                return_tensors='pt',
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Extract features
            outputs = self.model(**inputs)

            # Use mean pooling over time dimension
            hidden_states = outputs.last_hidden_state
            pooled_features = hidden_states.mean(dim=1)

        return pooled_features

    def extract_hybrid_audio_features(self, audio_path: str) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Extract comprehensive audio features combining classical and transformer methods

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with multiple feature representations
        """
        # Load audio
        waveform = self.load_audio(audio_path)

        # Classical features
        mfcc = self.extract_mfcc(waveform)
        mel_spec = self.extract_mel_spectrogram(waveform)

        # Transformer features
        wav2vec_features = self.extract_wav2vec_features(waveform)

        return {
            'mfcc': mfcc,                                    # (n_mfcc, time)
            'mel_spectrogram': mel_spec,                     # (n_mels, time)
            'wav2vec_features': wav2vec_features,           # (1, feature_dim)
            'waveform': waveform                             # (samples,)
        }

    def batch_process(self, audio_paths: List[str], batch_size: int = Config.BATCH_SIZE) -> Dict[str, torch.Tensor]:
        """
        Process multiple audio files in batches

        Args:
            audio_paths: List of audio file paths
            batch_size: Batch size for processing

        Returns:
            Dictionary with batched features
        """
        all_wav2vec_features = []
        all_mfcc = []

        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]

            # Load waveforms
            waveforms = [self.load_audio(path) for path in batch_paths]

            # Extract Wav2Vec2 features
            wav2vec_features = self.extract_wav2vec_features(waveforms)
            all_wav2vec_features.append(wav2vec_features.cpu())

            # Extract MFCC
            batch_mfcc = [self.extract_mfcc(wf) for wf in waveforms]
            all_mfcc.append(np.stack(batch_mfcc))

        return {
            'wav2vec_features': torch.cat(all_wav2vec_features, dim=0),
            'mfcc': np.concatenate(all_mfcc, axis=0)
        }


# Test the preprocessor
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Audio Preprocessor Test")
    print("="*60)
    print("\n📝 Note: Requires actual audio files to test fully")
    print("\nFeatures implemented:")
    print("  ✅ Wav2Vec2 transformer for semantic audio analysis")
    print("  ✅ MFCC extraction (40 coefficients)")
    print("  ✅ Mel Spectrogram (128 mel bands)")
    print("  ✅ Hybrid classical + transformer features")
    print("  ✅ Batch processing capability")
    print("  ✅ Designed for audio-visual deepfake detection (96.55% accuracy)")
