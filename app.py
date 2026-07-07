import os
import tempfile

import librosa
import matplotlib.pyplot as plt
import streamlit as st
from report_generator import generate_report
from speech_to_text import transcribe_audio
from semantic_eval import load_reference_concepts, calculate_similarity
from audio_utils import extract_audio_features
from scoring_engine import (
    calculate_filler_ratio,
    calculate_final_score,
    classify_understanding,
)



st.set_page_config(
    page_title="Voice-Based Concept Understanding Analyser",
    page_icon="🎤",
    layout="wide",
)



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




def plot_waveform(audio_path):

    signal, sr = librosa.load(audio_path, sr=None)

    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(signal)

    ax.set_title("Audio Waveform")

    ax.set_xlabel("Time")

    ax.set_ylabel("Amplitude")

    waveform_path = "waveform.png"

    plt.savefig(waveform_path)

    st.pyplot(fig)

    plt.close(fig)

    return waveform_path




left, right = st.columns([1, 2])

with left:

    st.subheader("📚 Reference Concept")

    concept_name = st.selectbox(
        "Choose a Concept",
        ["Machine Learning", "Cloud Computing", "Artificial Intelligence"],
    )

with right:

    st.subheader("🎵 Upload Audio")

    uploaded = st.file_uploader("Upload WAV Audio File", type=["wav"])

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

        st.session_state.transcript = ""
        st.session_state.similarity = 0.0
        st.session_state.features = {}
        st.session_state.filler_ratio = 0.0
        st.session_state.final_score = 0.0
        st.session_state.feedback = ""
        st.session_state.analysis_complete = False

        st.rerun()



    if analyze:

        try:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:

                tmp.write(uploaded.read())

                audio_path = tmp.name

            st.subheader("📈 Waveform Visualization")

            waveform_path = plot_waveform(audio_path)

            with st.spinner("Analyzing audio..."):

                transcript = transcribe_audio(audio_path)

                concepts = load_reference_concepts()

                reference = concepts[concept_name]

                similarity = calculate_similarity(reference, transcript)

                features = extract_audio_features(audio_path)

                filler_ratio = calculate_filler_ratio(transcript)

                final_score = calculate_final_score(
                    similarity,
                    filler_ratio,
                    features["rms_energy"],
                    features["pause_ratio"],
                )

                feedback = classify_understanding(final_score)

                pdf_file = generate_report(
                    filename="evaluation_report.pdf",
                    concept=reference,
                    transcript=transcript,
                    similarity=similarity,
                    filler_ratio=filler_ratio,
                    pause_ratio=features["pause_ratio"],
                    rms_energy=features["rms_energy"],
                    score=final_score,
                    feedback=feedback,
                    waveform_path=waveform_path,
                )

            st.session_state.transcript = transcript
            st.session_state.similarity = similarity
            st.session_state.features = features
            st.session_state.filler_ratio = filler_ratio
            st.session_state.final_score = final_score
            st.session_state.feedback = feedback
            st.session_state.pdf = pdf_file
            st.session_state.analysis_complete = True

        except Exception as e:

            st.error("Unable to process the uploaded audio.")

            st.exception(e)

        finally:

            if "audio_path" in locals() and os.path.exists(audio_path):
                os.remove(audio_path)



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

    st.subheader("📊 Evaluation Metrics")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric("Semantic Similarity", f"{st.session_state.similarity:.2f}")

    with c2:

        st.metric("Pause Ratio", st.session_state.features["pause_ratio"])

    with c3:

        st.metric("RMS Energy", st.session_state.features["rms_energy"])

    with c4:

        st.metric("Filler Ratio", st.session_state.filler_ratio)

    st.markdown("---")

    st.subheader("🎯 Final Evaluation")

    score1, score2 = st.columns(2)

    with score1:

        st.metric("Final Understanding Score", f"{st.session_state.final_score}/100")
        st.progress(st.session_state.final_score / 100)

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
if st.session_state.pdf:

    with open(st.session_state.pdf, "rb") as pdf:

        st.download_button(
            label="📄 Download Evaluation Report",
            data=pdf,
            file_name="evaluation_report.pdf",
            mime="application/pdf",
        )
