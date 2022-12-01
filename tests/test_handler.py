import pytest

from handler import WhisperHandler
from tests.utils import MockContext


@pytest.fixture()
def serve_context():
    context = MockContext(
        model_name="base_whisper",
        model_type="base.en",
        model_dir="model_pt",
    )
    return context


def initialize(serve_context):
    handler = WhisperHandler()
    handler.initialize(serve_context)

    return handler


def test_handle(serve_context):
    context = serve_context
    handler = initialize(serve_context)
    with open("tests/data/jfk.flac", "rb") as f:
        audio_bytes = f.read()
    test_data = [{"data": audio_bytes}]
    results = handler.handle(test_data, context)

    assert len(results) == 1

    transcription = results[0].lower()
    assert "my fellow americans" in transcription
    assert "your country" in transcription
    assert "do for you" in transcription


def test_handle_batch(serve_context):
    context = serve_context
    handler = initialize(serve_context)
    with open("tests/data/jfk.flac", "rb") as f:
        audio_bytes = f.read()

    test_data = [{"data": audio_bytes}] * 2
    results = handler.handle(test_data, context)
    assert len(results) == 2
    assert "my fellow americans" in results[1].lower()
