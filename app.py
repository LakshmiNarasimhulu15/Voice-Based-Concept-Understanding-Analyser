import os
import tempfile

import librosa
import matplotlib.pyplot as plt
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

# -----------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------

st.set_page_config(
    page_title="Voice-Based Concept Understanding Analyser",
    page_icon="🎤",
    layout="wide"
)

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------

st.sidebar.title("📘 About Project")

st.sidebar.info(
    """
Voice-Based Concept Understanding Analyser

This application evaluates:

• Speech Transcription (Whisper)

• Semantic Similarity (Sentence-BERT)

• Audio Features (Librosa)

• Understanding Score

• Communication Fluency
"""
)

st.sidebar.markdown("---")

st.sidebar.success("SmartBridge - Google Cloud Generative AI Internship")

# -----------------------------------------------------
# Main Title
# -----------------------------------------------------

st.title("🎤 Voice-Based Concept Understanding Analyser")

st.markdown(
"""
Evaluate spoken explanations using Artificial Intelligence.

The system analyzes:

- 🎙 Speech transcription
- 🧠 Concept understanding
- 📈 Semantic similarity
- 🔊 Audio quality
- ⭐ Overall understanding score
"""
)

st.markdown("---")

# -----------------------------------------------------
# Waveform Function
# -----------------------------------------------------

def plot_waveform(audio_path):

    signal, sample_rate = librosa.load(audio_path, sr=None)

    fig, ax = plt.subplots(figsize=(10,3))

    ax.plot(signal, color="royalblue")

    ax.set_title("Uploaded Audio Waveform")

    ax.set_xlabel("Samples")

    ax.set_ylabel("Amplitude")

    st.pyplot(fig)

    plt.close(fig)

# -----------------------------------------------------
# Input Layout
# -----------------------------------------------------

left, right = st.columns([1,2])

with left:

    st.subheader("📚 Reference Concept")

    concept_name = st.selectbox(
        "Choose a Concept",
        [
            "Machine Learning",
            "Cloud Computing",
            "Artificial Intelligence"
        ]
    )

with right:

    st.subheader("🎵 Upload Audio")

    uploaded = st.file_uploader(
        "Upload WAV Audio File",
        type=["wav"]
    )

# -----------------------------------------------------
# Audio Section
# -----------------------------------------------------

if uploaded:

    st.markdown("---")

    st.subheader("🔊 Audio Preview")

    st.audio(uploaded)

    if st.button("🚀 Analyze Audio"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp:

            tmp.write(uploaded.read())

            audio_path = tmp.name

        st.subheader("📈 Waveform Visualization")

        plot_waveform(audio_path)

        with st.spinner("Analyzing audio... Please wait..."):

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

        st.success("✅ Analysis Completed Successfully")

        st.markdown("---")

        # -------------------------------------------------
        # Transcript
        # -------------------------------------------------

        with st.expander("📝 View Transcript", expanded=True):

            st.text_area(
                "Speech Transcript",
                transcript,
                height=180
            )

        st.markdown("---")

        # -------------------------------------------------
        # Evaluation Metrics
        # -------------------------------------------------

        st.subheader("📊 Evaluation Metrics")

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Semantic Similarity",
                f"{similarity:.2f}"
            )

        with c2:

            st.metric(
                "Pause Ratio",
                features["pause_ratio"]
            )

        with c3:

            st.metric(
                "RMS Energy",
                features["rms_energy"]
            )

        with c4:

            st.metric(
                "Filler Ratio",
                filler_ratio
            )

        st.markdown("---")

        # -------------------------------------------------
        # Final Score
        # -------------------------------------------------

        st.subheader("🎯 Final Evaluation")

        score_col1, score_col2 = st.columns(2)

        with score_col1:

            st.metric(
                "Final Understanding Score",
                f"{final_score}/100"
            )

        with score_col2:

            if feedback == "Strong Understanding":

                st.success(feedback)

            elif feedback == "Moderate Understanding":

                st.warning(feedback)

            else:

                st.error(feedback)

        st.markdown("---")

        st.caption(
            "Voice-Based Concept Understanding Analyser | SmartBridge Internship Project"
        )

        os.remove(audio_path)