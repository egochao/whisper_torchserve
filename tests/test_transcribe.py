import whisper


def test_transcribe_base_en():
    model_name = "base.en"
    model = whisper.load_model(model_name, download_root="./model_pt")
    audio_path = "tests/data/jfk.flac"

    language = "en" if model_name.endswith(".en") else None
    result = model.transcribe(audio_path, language=language)
    assert result["language"] == "en"

    transcription = result["text"].lower()
    assert "my fellow americans" in transcription
    assert "your country" in transcription
    assert "do for you" in transcription
