
import re

# Filler words
FILLER_WORDS = [
    "um",
    "uh",
    "like",
    "actually",
    "basically",
    "so"
]


def calculate_filler_ratio(transcript):
    """
    Calculate filler word ratio.
    """

    words = re.findall(r"\b\w+\b", transcript.lower())

    if len(words) == 0:
        return 0.0

    filler_count = sum(
        1 for word in words if word in FILLER_WORDS
    )

    return round(filler_count / len(words), 3)


def calculate_score(similarity):
    """
    Backward compatibility function.
    """
    return int(similarity * 100)


def calculate_final_score(
    similarity,
    filler_ratio,
    rms_energy,
    pause_ratio
):
    """
    Combine all metrics into one final score.
    """

    semantic_score = similarity * 100

    filler_penalty = filler_ratio * 100

    pause_penalty = pause_ratio * 30

    confidence_bonus = rms_energy * 20

    score = (
        semantic_score
        - filler_penalty
        - pause_penalty
        + confidence_bonus
    )

    score = max(0, min(score, 100))

    return round(score, 2)


def classify_understanding(score):
    """
    Return qualitative feedback.
    """

    if score >= 80:
        return "Strong Understanding"

    elif score >= 60:
        return "Moderate Understanding"

    else:
        return "Poor Understanding"