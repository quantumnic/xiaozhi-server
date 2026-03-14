"""Tests for session state machine and TTS sentence streamer."""

from xiaozhi.tts import SentenceStreamer, split_sentences
from xiaozhi.session import SessionState


class TestSentenceStreamer:
    def test_single_sentence(self):
        s = SentenceStreamer()
        result = s.add("Hello world.")
        # May or may not yield depending on trailing space
        remaining = s.flush()
        total = result + ([remaining] if remaining else [])
        assert any("Hello world." in t for t in total)

    def test_multiple_sentences(self):
        s = SentenceStreamer()
        sentences = s.add("First sentence. Second sentence. ")
        assert len(sentences) >= 1

    def test_streaming_chunks(self):
        s = SentenceStreamer()
        all_sentences = []
        for chunk in ["Hello ", "world. ", "How are ", "you? ", "Fine."]:
            all_sentences.extend(s.add(chunk))
        remaining = s.flush()
        if remaining:
            all_sentences.append(remaining)
        assert len(all_sentences) >= 2

    def test_flush_empty(self):
        s = SentenceStreamer()
        assert s.flush() is None

    def test_flush_with_content(self):
        s = SentenceStreamer()
        s.add("incomplete sentence")
        result = s.flush()
        assert result == "incomplete sentence"


class TestSplitSentences:
    def test_basic(self):
        result = split_sentences("Hello. World. How are you?")
        assert len(result) >= 2

    def test_empty(self):
        assert split_sentences("") == []

    def test_single(self):
        result = split_sentences("Just one sentence.")
        assert len(result) == 1

    def test_chinese_punctuation(self):
        result = split_sentences("你好。世界！")
        assert len(result) >= 1


class TestSessionState:
    def test_states_exist(self):
        assert SessionState.IDLE == "idle"
        assert SessionState.LISTENING == "listening"
        assert SessionState.PROCESSING == "processing"
        assert SessionState.SPEAKING == "speaking"
