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
# Session State Initialization
# -----------------------------------------------------

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "similarity" not in st.session_state:
    st.session_state.similarity = 0.0

if "features" not in st.session_state:
    st.session_state.features = {}

if "filler_ratio" not in st.session_state:
    st.session_state.filler_ratio = 0.0

if "final_score" not in st.session_state:
    st.session_state.final_score = 0.0

if "feedback" not in st.session_state:
    st.session_state.feedback = ""

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------

st.sidebar.title("📘 About Project")

st.sidebar.info("""
Voice-Based Concept Understanding Analyser

This application evaluates:

• Speech Transcription (Whisper)

• Semantic Similarity (Sentence-BERT)

• Audio Features (Librosa)

• Understanding Score

• Communication Fluency
""")

st.sidebar.markdown("---")

st.sidebar.success("SmartBridge - Google Cloud Generative AI Internship")

# -----------------------------------------------------
# Main Title
# -----------------------------------------------------

st.title("🎤 Voice-Based Concept Understanding Analyser")

st.markdown("""
Evaluate spoken explanations using Artificial Intelligence.

The system analyzes:

- 🎙 Speech transcription
- 🧠 Concept understanding
- 📈 Semantic similarity
- 🔊 Audio quality
- ⭐ Overall understanding score
""")

st.markdown("---")

# -----------------------------------------------------
# Waveform Function
# -----------------------------------------------------

def plot_waveform(audio_path):

    signal, sr = librosa.load(audio_path, sr=None)

    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(signal, color="royalblue")

    ax.set_title("Uploaded Audio Waveform")

    ax.set_xlabel("Samples")

    ax.set_ylabel("Amplitude")

    st.pyplot(fig)

    plt.close(fig)

# -----------------------------------------------------
# Input Layout
# -----------------------------------------------------

left, right = st.columns([1, 2])

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

if uploaded is None:
    st.info("Please upload a WAV audio file to begin analysis.")

# -----------------------------------------------------
# Audio Section
# -----------------------------------------------------

if uploaded is not None:

    st.markdown("---")

    st.subheader("🔊 Audio Preview")

    st.audio(uploaded)

    col1, col2 = st.columns(2)

    with col1:
        analyze = st.button(
            "🚀 Analyze Audio",
            use_container_width=True
        )

    with col2:
        clear = st.button(
            "🗑 Clear Results",
            use_container_width=True
        )

    # ---------------------------------------------
    # Clear Session
    # ---------------------------------------------

    if clear:

        st.session_state.transcript = ""
        st.session_state.similarity = 0.0
        st.session_state.features = {}
        st.session_state.filler_ratio = 0.0
        st.session_state.final_score = 0.0
        st.session_state.feedback = ""
        st.session_state.analysis_complete = False

        st.rerun()

    # ---------------------------------------------
    # Analyze
    # ---------------------------------------------

    if analyze:

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as tmp:

                tmp.write(uploaded.read())

                audio_path = tmp.name

            st.subheader("📈 Waveform Visualization")

            plot_waveform(audio_path)

            with st.spinner("Analyzing audio..."):

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

            st.session_state.transcript = transcript
            st.session_state.similarity = similarity
            st.session_state.features = features
            st.session_state.filler_ratio = filler_ratio
            st.session_state.final_score = final_score
            st.session_state.feedback = feedback
            st.session_state.analysis_complete = True

        except Exception as e:

            st.error("Unable to process the uploaded audio.")

            st.exception(e)

        finally:

            if "audio_path" in locals() and os.path.exists(audio_path):
                os.remove(audio_path)

# -----------------------------------------------------
# Results
# -----------------------------------------------------

if st.session_state.analysis_complete:

    st.success("✅ Analysis Completed Successfully")

    st.markdown("---")

    with st.expander(
        "📝 View Transcript",
        expanded=True
    ):

        st.text_area(
            "Speech Transcript",
            st.session_state.transcript,
            height=180
        )

    st.markdown("---")

    st.subheader("📊 Evaluation Metrics")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Semantic Similarity",
            f"{st.session_state.similarity:.2f}"
        )

    with c2:

        st.metric(
            "Pause Ratio",
            st.session_state.features["pause_ratio"]
        )

    with c3:

        st.metric(
            "RMS Energy",
            st.session_state.features["rms_energy"]
        )

    with c4:

        st.metric(
            "Filler Ratio",
            st.session_state.filler_ratio
        )

    st.markdown("---")

    st.subheader("🎯 Final Evaluation")

    score1, score2 = st.columns(2)

    with score1:

        st.metric(
            "Final Understanding Score",
            f"{st.session_state.final_score}/100"
        )

    with score2:

        feedback = st.session_state.feedback

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