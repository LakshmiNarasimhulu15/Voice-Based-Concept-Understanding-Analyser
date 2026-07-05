import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_reference_concepts(file_path="concepts.json"):
    """Load reference concepts from JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def generate_embedding(text):
    """Generate Sentence-BERT embedding."""
    return model.encode(text)


def calculate_similarity(reference_text, student_text):
    """Calculate cosine similarity between reference and student text."""

    if not student_text.strip():
        return 0.0

    reference_embedding = generate_embedding(reference_text)
    student_embedding = generate_embedding(student_text)

    similarity = cosine_similarity(
        [reference_embedding],
        [student_embedding]
    )[0][0]

    return float(similarity)


def normalize_score(similarity):
    """
    Convert cosine similarity (-1 to 1)
    into a percentage (0 to 100).
    """

    normalized = ((similarity + 1) / 2) * 100

    return round(normalized, 2)


def evaluate_understanding(score):

    if score >= 85:
        return "Excellent Understanding"

    elif score >= 70:
        return "Good Understanding"

    elif score >= 55:
        return "Moderate Understanding"

    else:
        return "Poor Understanding"