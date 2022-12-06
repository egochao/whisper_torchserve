from detect_mispronunciation import detect_mispronunced_words

def test_detect_mispronunciation():
    target_text = "This is a target text to be compared with"
    test_text = "This is a test text to be compared with"
    assert detect_mispronunced_words(target_text, test_text) == {"target"}

