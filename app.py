import os
import tempfile

import streamlit as st

from speech_to_text import transcribe_audio

st.set_page_config(
    page_title="Voice-Based Concept Understanding Analyser",
    layout="wide"
)

st.title("🎤 Voice-Based Concept Understanding Analyser")

uploaded = st.file_uploader(
    "Upload WAV Audio",
    type=["wav"]
)

if uploaded:

    st.audio(uploaded)

    if st.button("Transcribe"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp:

            tmp.write(uploaded.read())

            audio_path = tmp.name

        with st.spinner("Transcribing..."):

            transcript = transcribe_audio(audio_path)

        st.subheader("Transcript")

        st.write(transcript)

        os.remove(audio_path)