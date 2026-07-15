import librosa
import numpy as np
from typing import Tuple, Dict, Union

def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to 16 kHz mono.
    Returns:
        Tuple[np.ndarray, int]: Audio time series and sampling rate.
        If loading fails, returns an empty array and 16000.
    """
    try:
        audio, sample_rate = librosa.load(
            file_path,
            sr=16000,
            mono=True
        )
        return audio, sample_rate
    except Exception:
        # Return empty numpy array and default sample rate on failure
        return np.array([], dtype=np.float32), 16000


def extract_features(file_path: str) -> Dict[str, float]:
    """
    Wrapper function to extract all audio features.
    """
    return extract_audio_features(file_path)


def extract_audio_features(audio_input: Union[str, np.ndarray, Tuple[np.ndarray, int]]) -> Dict[str, float]:
    """
    Extract audio-level features:
    - RMS Energy
    - Pause Ratio

    Args:
        audio_input: File path (str), NumPy array of audio waveform, or (y, sr) tuple.

    Returns:
        Dict[str, float]: Dictionary containing rms_energy and pause_ratio.
    """
    try:
        # Determine input format and get audio time series (y)
        if isinstance(audio_input, str):
            y, sr = load_audio(audio_input)
        elif isinstance(audio_input, tuple) and len(audio_input) == 2:
            y, sr = audio_input
        elif isinstance(audio_input, np.ndarray):
            y = audio_input
            sr = 16000
        else:
            return {"rms_energy": 0.0, "pause_ratio": 0.0}

        # Handle empty audio array gracefully
        if y is None or len(y) == 0:
            return {"rms_energy": 0.0, "pause_ratio": 0.0}

        # Calculate RMS energy
        rms = librosa.feature.rms(y=y)[0]

        if len(rms) == 0:
            return {"rms_energy": 0.0, "pause_ratio": 0.0}

        avg_rms = float(np.mean(rms))

        # Calculate Pause Ratio
        silence_threshold = 0.02
        silent_frames = np.sum(rms < silence_threshold)
        pause_ratio = silent_frames / len(rms)

        # Validate that values are not NaN or Inf
        if np.isnan(avg_rms) or np.isinf(avg_rms):
            avg_rms = 0.0
        if np.isnan(pause_ratio) or np.isinf(pause_ratio):
            pause_ratio = 0.0

        return {
            "rms_energy": round(avg_rms, 3),
            "pause_ratio": round(float(pause_ratio), 3)
        }
    except Exception:
        # Return safe defaults if any unexpected error occurs
        return {
            "rms_energy": 0.0,
            "pause_ratio": 0.0
        }