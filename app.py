import os
import tempfile

import streamlit as st

from speech_to_text import transcribe_audio
from semantic_eval import (
    load_reference_concepts,
    calculate_similarity
)
from audio_utils import extract_audio_features
from scoring_engine import (
    calculate_filler_ratio,
    calculate_final_score,
    classify_understanding
)

st.set_page_config(
    page_title="Voice-Based Concept Understanding Analyser",
    layout="wide"
)

st.title("🎤 Voice-Based Concept Understanding Analyser")

st.write(
    "Upload a WAV audio file to evaluate concept understanding."
)

uploaded = st.file_uploader(
    "Upload WAV Audio",
    type=["wav"]
)

if uploaded:

    st.audio(uploaded)

    concept_name = st.selectbox(
        "Select Reference Concept",
        [
            "Machine Learning",
            "Cloud Computing",
            "Artificial Intelligence"
        ]
    )

    if st.button("Analyze"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp:

            tmp.write(uploaded.read())

            audio_path = tmp.name

        with st.spinner("Processing..."):

            transcript = transcribe_audio(audio_path)

            concepts = load_reference_concepts()

            reference = concepts[concept_name]

            similarity = calculate_similarity(
                reference,
                transcript
            )

            features = extract_audio_features(audio_path)

            filler_ratio = calculate_filler_ratio(
                transcript
            )

            final_score = calculate_final_score(
                similarity,
                filler_ratio,
                features["rms_energy"],
                features["pause_ratio"]
            )

            feedback = classify_understanding(
                final_score
            )

        st.success("Analysis Complete")

        st.subheader("Transcript")

        st.write(transcript)

        st.subheader("Evaluation")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Semantic Similarity",
                f"{similarity:.2f}"
            )

            st.metric(
                "Pause Ratio",
                features["pause_ratio"]
            )

        with col2:

            st.metric(
                "RMS Energy",
                features["rms_energy"]
            )

            st.metric(
                "Filler Ratio",
                filler_ratio
            )

        st.metric(
            "Final Understanding Score",
            final_score
        )

        st.success(
            f"Classification: {feedback}"
        )

        os.remove(audio_path)