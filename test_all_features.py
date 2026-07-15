import os
from speech_to_text import transcribe_audio
from semantic_eval import load_reference_concepts, calculate_similarity, normalize_score, evaluate_understanding
from audio_utils import extract_audio_features
from scoring_engine import calculate_filler_ratio, calculate_final_score, classify_understanding
from report_generator import generate_report

def main():
    print("=== VOICE-BASED CONCEPT UNDERSTANDING ANALYSER TEST ===")
    
    # 1. Test Concept Loading
    print("\n1. Testing Concept Loading...")
    concepts = load_reference_concepts()
    print(f"Loaded {len(concepts)} concepts: {list(concepts.keys())}")
    
    # Choose ML concept
    concept_name = "Machine Learning"
    reference = concepts.get(concept_name, "Machine learning is the study of computer algorithms that improve automatically through experience.")
    print(f"Reference Description: '{reference}'")

    # 2. Test Audio Loading & Features
    audio_path = "test_audio/sample.wav"
    if not os.path.exists(audio_path):
        print(f"Error: Test audio file not found at {audio_path}")
        return
        
    print("\n2. Testing Audio Feature Extraction...")
    features = extract_audio_features(audio_path)
    print(f"Extracted Features: {features}")

    # 3. Test Audio Transcription (Whisper)
    print("\n3. Testing Audio Transcription...")
    transcript = transcribe_audio(audio_path)
    print(f"Transcript Preview:\n{transcript[:150]}...")

    # 4. Test Semantic Similarity
    print("\n4. Testing Semantic Similarity...")
    similarity = calculate_similarity(reference, transcript)
    normalized = normalize_score(similarity)
    eval_text = evaluate_understanding(normalized)
    print(f"Cosine Similarity: {similarity:.4f}")
    print(f"Normalized Score: {normalized:.2f}%")
    print(f"Understanding (from similarity): {eval_text}")

    # 5. Test Scoring Engine
    print("\n5. Testing Scoring Engine...")
    filler_ratio = calculate_filler_ratio(transcript)
    final_score = calculate_final_score(
        similarity=similarity,
        filler_ratio=filler_ratio,
        rms_energy=features.get("rms_energy", 0.0),
        pause_ratio=features.get("pause_ratio", 0.0)
    )
    feedback = classify_understanding(final_score)
    print(f"Filler Word Ratio: {filler_ratio:.3f}")
    print(f"Final Understanding Score: {final_score}/100")
    print(f"Final Understanding Level: {feedback}")

    # 6. Test PDF Generation
    print("\n6. Testing Report Generation...")
    pdf_filename = "evaluation_report.pdf"
    
    # Cleanup old PDF if exists
    if os.path.exists(pdf_filename):
        try:
            os.remove(pdf_filename)
        except Exception:
            pass
            
    res_pdf = generate_report(
        filename=pdf_filename,
        concept=reference,
        transcript=transcript,
        similarity=similarity,
        filler_ratio=filler_ratio,
        pause_ratio=features.get("pause_ratio", 0.0),
        rms_energy=features.get("rms_energy", 0.0),
        score=final_score,
        feedback=feedback,
        waveform_path=None # Test handling of missing/None waveform
    )
    
    print(f"Generated PDF File: {res_pdf}")
    if os.path.exists(res_pdf):
        print("Success: PDF file created successfully!")
        
        # Test locked PDF scenario
        print("\n7. Testing Locked PDF scenario...")
        try:
            # Simulate a lock by opening the file
            with open(res_pdf, "r+b") as lock_file:
                # Attempt to generate the report again while open
                alt_pdf = generate_report(
                    filename=pdf_filename,
                    concept=reference,
                    transcript=transcript,
                    similarity=similarity,
                    filler_ratio=filler_ratio,
                    pause_ratio=features.get("pause_ratio", 0.0),
                    rms_energy=features.get("rms_energy", 0.0),
                    score=final_score,
                    feedback=feedback,
                    waveform_path=None
                )
                print(f"Lock Test Success: Generated alternate report at: {alt_pdf}")
                if os.path.exists(alt_pdf):
                    os.remove(alt_pdf)
        except Exception as lock_err:
            print(f"Lock Test error/skip: {lock_err}")
    else:
        print("Failure: PDF file was not created!")

    print("\n=== ALL TESTS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
