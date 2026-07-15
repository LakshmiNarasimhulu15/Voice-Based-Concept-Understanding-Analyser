import os
import librosa
import whisper
import streamlit as st
import numpy as np
from audio_utils import load_audio

@st.cache_resource
def load_whisper_model():
    """
    Load Whisper model and cache it using Streamlit's cache_resource
    to ensure it only loads once.
    """
    return whisper.load_model("base")


def normalize_audio(input_path: str) -> np.ndarray:
    """
    Normalize audio:
    - Convert to mono
    - Resample to 16kHz
    Reuses load_audio from audio_utils.py to avoid duplicated logic.
    """
    audio_data, _ = load_audio(input_path)
    return audio_data


def transcribe_audio(audio_path: str) -> str:
    """
    Convert audio into text using Whisper by passing a preloaded
    NumPy array directly to bypass FFmpeg subprocess requirements.
    """
    if not audio_path or not os.path.exists(audio_path):
        return "Transcription Error: Audio file not found."

    try:
        # Load and normalize audio using the shared function
        audio_data = normalize_audio(audio_path)

        if audio_data is None or len(audio_data) == 0:
            return "Transcription Error: Invalid or empty audio file."

        # Load Whisper model lazily inside the function (uses Streamlit cache)
        model = load_whisper_model()

        # Transcribe directly from the NumPy array
        result = model.transcribe(
            audio_data,
            fp16=False
        )

        transcript = result.get("text", "").strip()

        if transcript == "":
            return "No speech detected."

        return transcript

    except Exception as e:
        return f"Transcription Error: {str(e)}"
