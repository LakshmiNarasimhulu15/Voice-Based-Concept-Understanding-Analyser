Prerequisites
-------------
- Python 3.10 or above

Steps
-----
1. Clone the repository.
2. Navigate to the project folder.
3. Create the virtual environment:
   python -m venv vbcu_env
4. Activate the environment.
5. Install dependencies:
   pip install -r requirements.txt
6. Verify the setup:
   python test_imports.py
7. Launch Streamlit:
   streamlit run app_test.py

   ## Project Structure

- app.py – Streamlit application
- audio_utils.py – Audio processing utilities
- speech_to_text.py – Speech transcription
- semantic_eval.py – Semantic similarity analysis
- scoring_engine.py – Final scoring logic
- report_generator.py – PDF report generation

Upload a mp3 file and click analyze button for analysis then see the statistics
