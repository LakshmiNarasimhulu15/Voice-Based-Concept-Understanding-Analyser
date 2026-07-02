import streamlit as st

from speech_to_text import transcribe_audio
from semantic_eval import calculate_similarity
from scoring_engine import calculate_score

st.set_page_config(
    page_title="Voice-Based Concept Understanding Analyser",
    layout="wide"
)

st.title("🎤 Voice-Based Concept Understanding Analyser")

st.write(
    "Upload an audio explanation to begin evaluation."
)

uploaded_file = st.file_uploader(
    "Choose an Audio File",
    type=["wav", "mp3", "m4a"]
)

if uploaded_file:

    st.success("Audio uploaded successfully!")

    st.audio(uploaded_file)

    if st.button("Analyze"):

        transcript = transcribe_audio(uploaded_file)

        similarity = calculate_similarity(
            "Reference Concept",
            transcript
        )

        score = calculate_score(similarity)

        st.subheader("Transcript")
        st.write(transcript)

        st.subheader("Semantic Similarity")
        st.write(similarity)

        st.subheader("Understanding Score")
        st.write(score)

        st.info("Waveform visualization will be added later.")

        st.info("PDF generation will be added later.")