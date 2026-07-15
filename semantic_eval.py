import os
import json
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@st.cache_resource
def load_semantic_model():
    """
    Load SentenceTransformer model and cache it using Streamlit's cache_resource
    to ensure it loads only once.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_model():
    """
    Get the SentenceTransformer model.
    Uses Streamlit caching if running in Streamlit, otherwise lazy-loads to a global.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            return load_semantic_model()
    except Exception:
        pass

    global _standalone_model
    if '_standalone_model' not in globals() or globals()['_standalone_model'] is None:
        globals()['_standalone_model'] = SentenceTransformer("all-MiniLM-L6-v2")
    return globals()['_standalone_model']


def load_reference_concepts(file_path="concepts.json"):
    """Load reference concepts from JSON file. Returns empty dict on failure."""
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}


def generate_embedding(text):
    """Generate Sentence-BERT embedding."""
    model = get_model()
    return model.encode(text)


def calculate_similarity(reference_text, student_text):
    """
    Calculate cosine similarity between reference and student text.
    Guarantees the returned value is a float between 0.0 and 1.0.
    """
    if reference_text is None or student_text is None:
        return 0.0

    ref_str = str(reference_text).strip()
    stud_str = str(student_text).strip()

    # If transcript is empty or no speech, return 0.0 similarity
    if not ref_str or not stud_str or stud_str == "No speech detected.":
        return 0.0

    try:
        reference_embedding = generate_embedding(ref_str)
        student_embedding = generate_embedding(stud_str)

        similarity = cosine_similarity(
            [reference_embedding],
            [student_embedding]
        )[0][0]

        val = float(similarity)

        # Guarantee returned value is float between 0.0 and 1.0
        val = max(0.0, min(val, 1.0))

        import math
        if math.isnan(val) or math.isinf(val):
            return 0.0

        return val
    except Exception:
        return 0.0


def normalize_score(similarity):
    """
    Convert cosine similarity (-1 to 1)
    into a percentage (0 to 100).
    """
    # Guarantee float similarity is valid
    try:
        sim_val = max(-1.0, min(float(similarity), 1.0))
        normalized = ((sim_val + 1) / 2) * 100
        return round(normalized, 2)
    except Exception:
        return 0.0


def evaluate_understanding(score):
    try:
        score_val = float(score)
    except Exception:
        score_val = 0.0

    if score_val >= 85:
        return "Excellent Understanding"
    elif score_val >= 70:
        return "Good Understanding"
    elif score_val >= 55:
        return "Moderate Understanding"
    else:
        return "Poor Understanding"
