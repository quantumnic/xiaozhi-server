"""Tests for audio utilities."""

import wave
import io
import numpy as np

from xiaozhi.audio import (
    AudioBuffer,
    OpusDecoder,
    OpusEncoder,
    pcm_to_wav,
    resample,
    split_pcm_frames,
)


class TestPcmToWav:
    def test_produces_valid_wav(self):
        # 100 samples of silence
        pcm = b"\x00\x00" * 100
        wav_data = pcm_to_wav(pcm, sample_rate=16000, channels=1)
        buf = io.BytesIO(wav_data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 100

    def test_stereo(self):
        pcm = b"\x00\x00" * 200  # 100 frames * 2 channels
        wav_data = pcm_to_wav(pcm, sample_rate=44100, channels=2)
        buf = io.BytesIO(wav_data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 2
            assert wf.getframerate() == 44100


class TestResample:
    def test_same_rate_noop(self):
        pcm = b"\x01\x00" * 100
        result = resample(pcm, 16000, 16000)
        assert result == pcm

    def test_upsample(self):
        # 160 samples at 16kHz → should become ~240 samples at 24kHz
        pcm = np.zeros(160, dtype=np.int16).tobytes()
        result = resample(pcm, 16000, 24000)
        result_samples = len(result) // 2
        assert result_samples == 240

    def test_downsample(self):
        pcm = np.zeros(480, dtype=np.int16).tobytes()
        result = resample(pcm, 48000, 16000)
        result_samples = len(result) // 2
        assert result_samples == 160


class TestSplitPcmFrames:
    def test_exact_split(self):
        # 60ms at 24kHz = 1440 samples = 2880 bytes
        frame_bytes = 1440 * 2
        pcm = b"\x00\x00" * (1440 * 3)  # 3 frames
        frames = split_pcm_frames(pcm, 60, 24000)
        assert len(frames) == 3
        assert all(len(f) == frame_bytes for f in frames)

    def test_last_frame_padded(self):
        # 2.5 frames worth
        pcm = b"\x00\x00" * (1440 * 2 + 720)
        frames = split_pcm_frames(pcm, 60, 24000)
        assert len(frames) == 3
        assert len(frames[2]) == 1440 * 2  # padded to full frame

    def test_empty_input(self):
        frames = split_pcm_frames(b"", 60, 24000)
        assert frames == []


class TestOpusCodecFallback:
    """Test that Opus encoder/decoder work in fallback mode (raw PCM pass-through)."""

    def test_decoder_fallback(self):
        dec = OpusDecoder.__new__(OpusDecoder)
        dec.sample_rate = 16000
        dec.channels = 1
        dec._decoder = None  # Force fallback
        data = b"\x01\x02\x03\x04"
        assert dec.decode(data) == data

    def test_encoder_fallback(self):
        enc = OpusEncoder.__new__(OpusEncoder)
        enc.sample_rate = 24000
        enc.channels = 1
        enc._encoder = None  # Force fallback
        data = b"\x01\x02\x03\x04"
        assert enc.encode(data) == data


class TestAudioBuffer:
    def test_add_and_count(self):
        buf = AudioBuffer(sample_rate=16000, channels=1, frame_duration=60)
        assert buf.frame_count == 0
        buf.add_frame(b"\x00" * 100)
        buf.add_frame(b"\x00" * 100)
        assert buf.frame_count == 2

    def test_clear(self):
        buf = AudioBuffer()
        buf.add_frame(b"\x00" * 100)
        buf.clear()
        assert buf.frame_count == 0

    def test_get_pcm_concatenates(self):
        buf = AudioBuffer()
        buf._frame_size = 960  # ensure frame_size is set
        buf.decoder._decoder = None  # force fallback (passthrough)
        frame1 = b"\x01\x00" * 10
        frame2 = b"\x02\x00" * 10
        buf.add_frame(frame1)
        buf.add_frame(frame2)
        pcm = buf.get_pcm()
        assert pcm == frame1 + frame2

    def test_get_wav(self):
        buf = AudioBuffer(sample_rate=16000)
        buf.decoder._decoder = None
        pcm = b"\x00\x00" * 160  # 10ms at 16kHz
        buf.add_frame(pcm)
        wav = buf.get_wav()
        assert wav[:4] == b"RIFF"
