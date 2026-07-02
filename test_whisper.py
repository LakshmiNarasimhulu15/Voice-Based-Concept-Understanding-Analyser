from speech_to_text import transcribe_audio

audio = "test_audio/sample.wav"

text = transcribe_audio(audio)

print()

print("Transcript:")

print(text)