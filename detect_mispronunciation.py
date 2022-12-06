from whisper.normalizers import EnglishTextNormalizer
from whisper.normalizers.english import EnglishNumberNormalizer, EnglishSpellingNormalizer
import constants

def detect_mispronunced_words(target_text, pronounced_text):
    """detect mispronunced words in test_text

    Args:
        target_text (str): target text
        test_text (str): test text

    Returns:
        mispronounced_words(set): set of mispronounced words
    """
    target_text = EnglishTextNormalizer()(target_text)
    pronounced_text = EnglishTextNormalizer()(pronounced_text)
    target_text = EnglishNumberNormalizer()(target_text)
    pronounced_text = EnglishNumberNormalizer()(pronounced_text)
    target_text = EnglishSpellingNormalizer()(target_text)
    pronounced_text = EnglishSpellingNormalizer()(pronounced_text)
    target_text = _remove_stop_words(target_text)
    pronounced_text = _remove_stop_words(pronounced_text)

    target_text = target_text.split()
    pronounced_text = pronounced_text.split()
    mispronounced_words = set([word for word in target_text if word not in pronounced_text])
    
    return mispronounced_words

def _remove_stop_words(text):
    """remove stop words from text

    Args:
        text (str): text to remove stop words from

    Returns:
        text (str): text without stop words
    """
    stop_words = constants.STOP_WORDS
    text = [word for word in text.split() if word not in stop_words]
    text = " ".join(text)
    return text
