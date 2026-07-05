
import librosa
import numpy as np


def load_audio(file_path):
    """
    Load audio file and resample to 16 kHz mono.
    """
    audio, sample_rate = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    return audio, sample_rate


def extract_features(file_path):
    """
    Wrapper function to extract all audio features.
    """
    return extract_audio_features(file_path)


def extract_audio_features(audio_path):
    """
    Extract audio-level features:
    - RMS Energy
    - Pause Ratio
    """

    y, sr = load_audio(audio_path)

    rms = librosa.feature.rms(y=y)[0]

    avg_rms = float(np.mean(rms))

    silence_threshold = 0.02

    silent_frames = np.sum(rms < silence_threshold)

    pause_ratio = silent_frames / len(rms)

    return {
        "rms_energy": round(avg_rms, 3),
        "pause_ratio": round(float(pause_ratio), 3)
    }