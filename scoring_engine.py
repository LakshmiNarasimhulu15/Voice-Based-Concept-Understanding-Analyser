import re
import math

FILLER_WORDS = [
    "um",
    "uh",
    "like",
    "actually",
    "basically",
    "so"
]


def _safe_float(val, default=0.0) -> float:
    """Helper to convert values to float safely, returning default on None/NaN/Inf/Error."""
    if val is None:
        return default
    try:
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return default
        return f_val
    except Exception:
        return default


def calculate_filler_ratio(transcript: str) -> float:
    """
    Calculate filler word ratio.
    """
    if not transcript or not isinstance(transcript, str):
        return 0.0

    words = re.findall(r"\b\w+\b", transcript.lower())

    if len(words) == 0:
        return 0.0

    filler_count = sum(
        1 for word in words if word in FILLER_WORDS
    )

    return round(filler_count / len(words), 3)


def calculate_score(similarity: float) -> int:
    """
    Backward compatibility function.
    """
    sim_val = _safe_float(similarity, 0.0)
    return int(sim_val * 100)


def calculate_final_score(
    similarity,
    filler_ratio,
    rms_energy,
    pause_ratio
) -> float:
    """
    Combine all metrics into one final score.
    """
    sim_val = _safe_float(similarity, 0.0)
    filler_val = _safe_float(filler_ratio, 0.0)
    rms_val = _safe_float(rms_energy, 0.0)
    pause_val = _safe_float(pause_ratio, 0.0)

    # Calculate subscores and penalties
    semantic_score = sim_val * 100
    filler_penalty = filler_val * 100
    pause_penalty = pause_val * 30
    confidence_bonus = rms_val * 20

    score = (
        semantic_score
        - filler_penalty
        - pause_penalty
        + confidence_bonus
    )

    # Clamp final score between 0 and 100
    score = max(0.0, min(score, 100.0))

    return round(score, 2)


def classify_understanding(score) -> str:
    """
    Return qualitative feedback.
    """
    score_val = _safe_float(score, 0.0)

    if score_val >= 80:
        return "Strong Understanding"
    elif score_val >= 60:
        return "Moderate Understanding"
    else:
        return "Poor Understanding"
