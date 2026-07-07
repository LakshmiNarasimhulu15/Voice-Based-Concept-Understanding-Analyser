import os
import tempfile

import librosa
import soundfile as sf
import whisper


MODEL = whisper.load_model("base")


def normalize_audio(input_path):
    """
    Normalize audio:
    - Convert to mono
    - Resample to 16kHz
    - Save temporary WAV
    """

    audio, sr = librosa.load(
        input_path,
        sr=16000,
        mono=True
    )

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    )

    sf.write(
        temp_file.name,
        audio,
        16000
    )

    return temp_file.name


def transcribe_audio(audio_path):
    """
    Convert audio into text.
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            "Audio file not found."
        )

    normalized = normalize_audio(audio_path)

    try:

        result = MODEL.transcribe(
            normalized,
            fp16=False
        )

        transcript = result["text"].strip()

        if transcript == "":
            return "No speech detected."

        return transcript

    except Exception as e:

        return f"Transcription Error: {str(e)}"

    finally:

        if os.path.exists(normalized):
            os.remove(normalized)
