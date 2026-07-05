from semantic_eval import (
    load_reference_concepts,
    calculate_similarity,
    normalize_score,
    evaluate_understanding
)

concepts = load_reference_concepts()

reference = concepts["Machine Learning"]

student = """
Machine learning allows computers to learn from data
without being directly programmed.
"""

similarity = calculate_similarity(reference, student)

score = normalize_score(similarity)

feedback = evaluate_understanding(score)

print("Reference:")
print(reference)

print("\nStudent:")
print(student)

print("\nSimilarity:", similarity)
print("Normalized Score:", score)
print("Feedback:", feedback)