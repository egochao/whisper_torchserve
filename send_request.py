import requests
import time

def main():
    with open("tests/jfk.flac", "rb") as f:
        audio_bytes = f.read()

    st = time.perf_counter()
    res = requests.post(
        "http://localhost:8888/predictions/whisper_base", 
        files={"data": audio_bytes})
    print(f"Time: {time.perf_counter() - st}")
    transcription =  res.text.lower()
    print(transcription)
    assert res.status_code == 200
    assert "my fellow americans" in transcription
    assert "your country" in transcription
    assert "do for you" in transcription

if __name__ == "__main__":
    main()