import os
import tempfile
import time
import librosa
import matplotlib.pyplot as plt
import streamlit as st
import soundfile as sf

from report_generator import generate_report
from speech_to_text import transcribe_audio
from semantic_eval import load_reference_concepts, calculate_similarity
from audio_utils import extract_audio_features
from scoring_engine import (
    calculate_filler_ratio,
    calculate_final_score,
    classify_understanding,
)

# Page configuration
st.set_page_config(
    page_title="Voice-Based Concept Understanding Analyser",
    page_icon="🎤",
    layout="wide",
)

# Helper to reset session state when inputs change
def reset_results():
    st.session_state.transcript = ""
    st.session_state.pdf = ""
    st.session_state.similarity = 0.0
    st.session_state.features = {}
    st.session_state.filler_ratio = 0.0
    st.session_state.final_score = 0.0
    st.session_state.feedback = ""
    st.session_state.analysis_complete = False
    st.session_state.transcription_time = 0.0
    st.session_state.similarity_time = 0.0
    st.session_state.feature_time = 0.0
    st.session_state.total_time = 0.0

# Initialize Session State
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "pdf" not in st.session_state:
    st.session_state.pdf = ""

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

if "transcription_time" not in st.session_state:
    st.session_state.transcription_time = 0.0

if "similarity_time" not in st.session_state:
    st.session_state.similarity_time = 0.0

if "feature_time" not in st.session_state:
    st.session_state.feature_time = 0.0

if "total_time" not in st.session_state:
    st.session_state.total_time = 0.0

# Sidebar Info
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

# Main Title and Description
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


def plot_waveform(audio_path: str) -> str:
    """
    Plot audio waveform and save as a unique temporary PNG file.
    Returns:
        str: Path to the generated image file, or None if failed.
    """
    try:
        signal, sr = librosa.load(audio_path, sr=None)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(signal, color="#1f77b4")
        ax.set_title("Audio Waveform")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        
        # Save to a unique temporary file
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        waveform_path = tmp_img.name
        tmp_img.close()
        
        fig.savefig(waveform_path, bbox_inches='tight')
        st.pyplot(fig)
        plt.close(fig)
        return waveform_path
    except Exception as e:
        st.warning(f"Could not generate waveform visualization: {e}")
        return None


# Layout
left, right = st.columns([1, 2])

# Load dynamic concepts
concepts = load_reference_concepts()
if not concepts:
    # Safe fallback inside UI if load fails
    concepts = {
        "Machine Learning": "Machine learning is a branch of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
        "Cloud Computing": "Cloud computing provides computing resources such as servers, storage, databases, networking, and software over the internet.",
        "Artificial Intelligence": "Artificial intelligence is the simulation of human intelligence processes by machines."
    }

concept_options = list(concepts.keys())

with left:
    st.subheader("📚 Reference Concept")
    concept_name = st.selectbox(
        "Choose a Concept",
        concept_options,
        on_change=reset_results
    )

with right:
    st.subheader("🎵 Upload Audio")
    uploaded = st.file_uploader(
        "Upload WAV Audio File", 
        type=["wav"],
        on_change=reset_results
    )

if uploaded is None:
    st.info("Please upload a WAV audio file to begin analysis.")

if uploaded is not None:
    st.markdown("---")
    st.subheader("🔊 Audio Preview")
    st.audio(uploaded)

    col1, col2 = st.columns(2)
    with col1:
        analyze = st.button("🚀 Analyze Audio", use_container_width=True)
    with col2:
        clear = st.button("🗑 Clear Results", use_container_width=True)

    if clear:
        reset_results()
        st.rerun()

    if analyze:
        audio_path = None
        waveform_path = None
        try:
            # Write uploaded file content to temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded.read())
                audio_path = tmp.name

            # Input validation: check file size
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                st.error("Uploaded audio file is empty or missing. Please upload a valid WAV file.")
                st.stop()

            # Input validation: verify WAV format and duration
            try:
                info = sf.info(audio_path)
                if info.duration == 0:
                    st.error("Uploaded audio file contains no audible sound (zero duration).")
                    st.stop()
            except Exception as sf_err:
                st.error(f"Uploaded file is corrupted or not a valid WAV audio file: {sf_err}")
                st.stop()

            # Generate Waveform Visualisation
            waveform_path = plot_waveform(audio_path)

            with st.spinner("Analyzing audio..."):
                overall_start = time.time()

                # 1. Speech Transcription
                start = time.time()
                transcript = transcribe_audio(audio_path)
                transcription_time = time.time() - start

                if "Transcription Error" in transcript:
                    st.error(transcript)
                    st.stop()

                # 2. Semantic Similarity
                reference = concepts.get(concept_name, "")
                if not reference:
                    st.error(f"Error: Concept description for '{concept_name}' not found.")
                    st.stop()

                start = time.time()
                similarity = calculate_similarity(reference, transcript)
                similarity_time = time.time() - start

                # 3. Audio Features
                start = time.time()
                features = extract_audio_features(audio_path)
                feature_time = time.time() - start

                # 4. Scoring Calculations
                filler_ratio = calculate_filler_ratio(transcript)
                total_time = time.time() - overall_start

                final_score = calculate_final_score(
                    similarity,
                    filler_ratio,
                    features.get("rms_energy", 0.0),
                    features.get("pause_ratio", 0.0),
                )

                feedback = classify_understanding(final_score)

                # 5. Report Generation
                pdf_file = generate_report(
                    filename="evaluation_report.pdf",
                    concept=reference,
                    transcript=transcript,
                    similarity=similarity,
                    filler_ratio=filler_ratio,
                    pause_ratio=features.get("pause_ratio", 0.0),
                    rms_energy=features.get("rms_energy", 0.0),
                    score=final_score,
                    feedback=feedback,
                    waveform_path=waveform_path,
                )

            # Store result values in session state
            st.session_state.transcript = transcript
            st.session_state.similarity = similarity
            st.session_state.features = features
            st.session_state.filler_ratio = filler_ratio
            st.session_state.final_score = final_score
            st.session_state.feedback = feedback
            st.session_state.transcription_time = transcription_time
            st.session_state.similarity_time = similarity_time
            st.session_state.feature_time = feature_time
            st.session_state.total_time = total_time
            st.session_state.pdf = pdf_file
            st.session_state.analysis_complete = True

        except Exception as e:
            st.error(f"An unexpected error occurred while processing the audio: {e}")
        finally:
            # Clean up temporary WAV audio file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
            # Clean up temporary PNG plot file
            if waveform_path and os.path.exists(waveform_path):
                try:
                    os.remove(waveform_path)
                except Exception:
                    pass

# Display results if analysis is complete
if st.session_state.analysis_complete:
    st.success("✅ Analysis Completed Successfully")

    st.info(f"""
### 📋 Evaluation Summary

**Reference Concept :** {concept_name}

**Understanding Level :** {st.session_state.feedback}

**Overall Score :** {st.session_state.final_score}/100
""")

    st.markdown("---")

    with st.expander("📝 View Transcript", expanded=True):
        st.text_area("Speech Transcript", st.session_state.transcript, height=180)

    st.markdown("---")
    st.subheader("⚡ Performance Metrics")

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("Transcription", f"{st.session_state.transcription_time:.2f} s")
    with p2:
        st.metric("Similarity", f"{st.session_state.similarity_time:.2f} s")
    with p3:
        st.metric("Feature Extraction", f"{st.session_state.feature_time:.2f} s")
    with p4:
        st.metric("Total Runtime", f"{st.session_state.total_time:.2f} s")

    st.markdown("---")
    st.subheader("📊 Evaluation Metrics")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Semantic Similarity", f"{st.session_state.similarity:.2f}")
    with c2:
        st.metric("Pause Ratio", st.session_state.features.get("pause_ratio", 0.0))
    with c3:
        st.metric("RMS Energy", st.session_state.features.get("rms_energy", 0.0))
    with c4:
        st.metric("Filler Ratio", st.session_state.filler_ratio)

    st.markdown("---")
    st.subheader("🎯 Final Evaluation")

    score1, score2 = st.columns(2)
    with score1:
        st.metric("Final Understanding Score", f"{st.session_state.final_score}/100")
        st.progress(st.session_state.final_score / 100.0)

    with score2:
        feedback = st.session_state.feedback
        if feedback == "Strong Understanding":
            st.success("🟢 Strong Understanding")
        elif feedback == "Moderate Understanding":
            st.warning("🟡 Moderate Understanding")
        else:
            st.error("🔴 Poor Understanding")

    st.markdown("---")
    st.caption(
        "Voice-Based Concept Understanding Analyser | SmartBridge Internship Project"
    )

if st.session_state.pdf and os.path.exists(st.session_state.pdf):
    with open(st.session_state.pdf, "rb") as pdf:
        st.download_button(
            label="📄 Download Evaluation Report",
            data=pdf,
            file_name=os.path.basename(st.session_state.pdf),
            mime="application/pdf",
        )
