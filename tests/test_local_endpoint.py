import pytest
import requests

@pytest.mark.integration
def test_send_real_request(start2serve):
    with open("tests/data/jfk.flac", "rb") as f:
        audio_bytes = f.read()

    res = requests.post(
        "http://localhost:8888/predictions/whisper_base", 
        files={"data": audio_bytes})
    assert res.status_code == 200
    transcription =  res.text.lower()
    assert "my fellow americans" in transcription
    assert "your country" in transcription
    assert "do for you" in transcription
