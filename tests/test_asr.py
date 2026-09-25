"""Tests for the speech layer: DSP, noise suppression, VAD, decoding, providers."""

from __future__ import annotations

import math
import random
import unittest

from tests.base import GenieTestCase, synth_audio


def tone(freq: float, seconds: float = 1.0, sample_rate: int = 16000,
         amplitude: float = 0.4):
    return [amplitude * math.sin(2 * math.pi * freq * i / sample_rate)
            for i in range(int(seconds * sample_rate))]


def syllable_train(seed: int, words: int = 6, sample_rate: int = 16000):
    """A synthetic 'speech-like' signal: a harmonic stack modulated into syllables."""
    rng = random.Random(seed)
    out = []
    f0 = 115 + rng.random() * 35
    for _ in range(words):
        duration = 0.28
        for i in range(int(duration * sample_rate)):
            t = i / sample_rate
            f0t = f0 * (1 + 0.05 * math.sin(2 * math.pi * 2.2 * t))
            signal = sum(math.sin(2 * math.pi * k * f0t * t) / (k ** 1.1)
                         for k in range(1, 20))
            envelope = max(0.0, math.sin(2 * math.pi * 2.6 * t)) ** 0.5
            out.append(0.22 * signal * envelope)
        out.extend([0.0] * int(sample_rate * 0.12))
    return out


class DspTests(GenieTestCase):
    def test_fft_matches_the_direct_transform(self):
        from interviewgenie.asr.dsp import fft

        rng = random.Random(4)
        samples = [rng.uniform(-1, 1) for _ in range(32)]
        real, imag = fft(samples)

        def dft(x, k):
            n = len(x)
            sr = sum(x[t] * math.cos(-2 * math.pi * k * t / n) for t in range(n))
            si = sum(x[t] * math.sin(-2 * math.pi * k * t / n) for t in range(n))
            return sr, si

        for k in (0, 1, 5, 17, 31):
            sr, si = dft(samples, k)
            self.assertAlmostEqual(real[k], sr, places=9)
            self.assertAlmostEqual(imag[k], si, places=9)

    def test_ifft_round_trips(self):
        from interviewgenie.asr.dsp import fft, ifft

        rng = random.Random(5)
        samples = [rng.uniform(-1, 1) for _ in range(64)]
        real, imag = fft(samples)
        back_re, _back_im = ifft(real, imag)
        for a, b in zip(samples, back_re):
            self.assertAlmostEqual(a, b, places=9)

    def test_mel_filterbank_is_normalised_and_overlapping(self):
        from interviewgenie.asr.dsp import mel_filterbank

        filters = mel_filterbank(26, 512, 16000)
        self.assertEqual(len(filters), 26)
        self.assertEqual(len(filters[0]), 257)
        for weights in filters:
            self.assertAlmostEqual(sum(weights), 1.0, places=6)
        # neighbouring filters overlap
        overlap = sum(a * b for a, b in zip(filters[0], filters[1]))
        self.assertGreater(overlap, 0.0)

    def test_mfcc_extractor_produces_consistent_frames(self):
        from interviewgenie.asr.dsp import Audio, MfccConfig, MfccExtractor

        extractor = MfccExtractor(MfccConfig(sample_rate=16000))
        first = extractor.extract(Audio(tone(220, 1.0), 16000))
        second = extractor.extract(Audio(tone(220, 1.0), 16000))
        self.assertTrue(first)
        self.assertEqual(len(first[0]), extractor.feature_dim)
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, places=9)

    def test_cmvn_centres_the_first_coefficient(self):
        from interviewgenie.asr.dsp import Audio, MfccConfig, MfccExtractor

        extractor = MfccExtractor(MfccConfig(sample_rate=16000, cmvn=True))
        features = extractor.extract(Audio(tone(440, 1.0), 16000))
        mean = sum(f[0] for f in features) / len(features)
        self.assertAlmostEqual(mean, 0.0, places=6)

    def test_resample_changes_length_as_expected(self):
        from interviewgenie.asr.dsp import resample

        samples = [0.0, 1.0, 0.0, -1.0] * 100
        down = resample(samples, 16000, 8000)
        self.assertEqual(len(down), len(samples) // 2)

    def test_pcm_round_trip(self):
        from interviewgenie.asr.dsp import int16_from_floats, pcm16_to_floats

        samples = [0.0, 0.5, -0.5, 1.0, -1.0]
        decoded = [v / 32768.0 for v in pcm16_to_floats(int16_from_floats(samples))]
        for a, b in zip(samples, decoded):
            self.assertAlmostEqual(a, b, places=4)


class NoiseSuppressionTests(GenieTestCase):
    def test_suppression_improves_measured_snr(self):
        from interviewgenie.asr.denoise import NoiseSuppressor, add_noise, measure_snr

        for kind in ("white", "pink", "babble", "hum"):
            for snr in (10, 5, 0):
                clean = syllable_train(3)
                noisy = add_noise(type("A", (), {
                    "samples": clean, "sample_rate": 16000, "channels": 1,
                    "label": "x"})(), snr, kind, seed=11)
                suppressor = NoiseSuppressor(sample_rate=16000, enabled=True)
                enhanced = suppressor.process(noisy.samples)
                before = measure_snr(noisy)
                after = measure_snr(type("A", (), {
                    "samples": enhanced, "sample_rate": 16000, "channels": 1,
                    "label": "y"})())
                self.assertGreater(after, before,
                                   msg=f"{kind}@{snr}dB: {before} -> {after}")

    def test_disabled_suppressor_is_a_no_op(self):
        from interviewgenie.asr.denoise import NoiseSuppressor

        suppressor = NoiseSuppressor(enabled=False)
        samples = tone(300, 0.3)
        self.assertEqual(suppressor.process(samples), samples)

    def test_add_noise_respects_the_requested_snr(self):
        from interviewgenie.asr.denoise import add_noise
        from interviewgenie.asr.dsp import Audio

        clean = Audio(tone(300, 1.0), 16000)
        for snr in (0, 10, 20):
            noisy = add_noise(clean, snr, "white", seed=1)
            power = sum(s * s for s in clean.samples) / len(clean.samples)
            noise = sum((n - c) ** 2 for n, c in zip(noisy.samples, clean.samples))
            noise /= len(clean.samples)
            self.assertAlmostEqual(10 * math.log10(power / noise), snr, delta=1.0)

    def test_silence_has_no_speech(self):
        from interviewgenie.asr.denoise import measure_snr
        from interviewgenie.asr.dsp import Audio

        silence = Audio([0.0] * 16000, 16000)
        self.assertGreaterEqual(measure_snr(silence), 50.0)


class VadTests(GenieTestCase):
    def test_speech_is_found_and_silence_is_not(self):
        from interviewgenie.asr.vad import VoiceActivityDetector
        from interviewgenie.asr.dsp import Audio

        detector = VoiceActivityDetector()
        samples = [0.0] * 4800 + syllable_train(3, 4) + [0.0] * 4800
        segments = detector.detect(Audio(samples, 16000))
        self.assertEqual(len(segments), 1)
        self.assertGreater(segments[0].duration, 0.6)

        self.assertEqual(detector.detect(Audio([0.0] * 16000, 16000)), [])

    def test_vad_survives_heavy_noise(self):
        from interviewgenie.asr.denoise import add_noise
        from interviewgenie.asr.dsp import Audio
        from interviewgenie.asr.vad import VoiceActivityDetector

        detector = VoiceActivityDetector()
        clean = Audio([0.0] * 4800 + syllable_train(3, 4) + [0.0] * 4800, 16000)
        for kind in ("white", "babble", "hum"):
            noisy = add_noise(clean, 0, kind, seed=5)
            segments = detector.detect(noisy)
            self.assertGreaterEqual(len(segments), 1, msg=kind)

    def test_noise_floor_does_not_climb_into_speech(self):
        from interviewgenie.asr.vad import VoiceActivityDetector
        from interviewgenie.asr.dsp import Audio

        detector = VoiceActivityDetector()
        detector.detect(Audio(syllable_train(9, 8), 16000))
        self.assertLess(detector._noise_floor, 0.01)


class DecoderTests(GenieTestCase):
    def test_dtw_finds_the_alignment(self):
        from interviewgenie.asr.decoder import dtw_cost

        a = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        b = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        cost, path = dtw_cost(a, b)
        self.assertLess(cost, 1e-9)
        self.assertTrue(path)

    def test_dtw_is_symmetric_for_identical_sequences(self):
        from interviewgenie.asr.decoder import dtw_cost

        seq = [[float(i), float(i * 2)] for i in range(10)]
        forward, _ = dtw_cost(seq, seq)
        self.assertLess(forward, 1e-9)

    def test_decoder_ranks_the_correct_phrase_first(self):
        from interviewgenie.asr.decoder import PhraseDecoder
        from interviewgenie.asr.templates import TemplateBank

        bank = TemplateBank()
        target = "tell me about yourself"
        decoy = "what is your greatest weakness"
        bank.add(target, synth_audio(target), source="generated", confidence=0.9)
        bank.add(decoy, synth_audio(decoy), source="generated", confidence=0.9)
        decoder = PhraseDecoder(bank=bank)
        result = decoder.decode(bank.extractor.extract(synth_audio(target)))
        self.assertIsNotNone(result)
        self.assertEqual(result.phrase, target)
        self.assertGreater(result.confidence, 0.0)

    def test_template_bank_persists(self):
        import os
        import tempfile

        from interviewgenie.asr.templates import TemplateBank

        bank = TemplateBank()
        bank.add("hello there", synth_audio("hello there"), confidence=0.9)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bank.json")
            bank.save(path)
            loaded = TemplateBank.load(path)
            self.assertEqual(loaded.phrases, bank.phrases)

    def test_template_averaging(self):
        from interviewgenie.asr.templates import TemplateBank, average_templates

        bank = TemplateBank()
        for _ in range(3):
            bank.add("same phrase", synth_audio("same phrase"))
        merged = average_templates(bank.templates)
        self.assertIsNotNone(merged)
        self.assertEqual(merged.source, "averaged")
        self.assertEqual(merged.samples, 3)


class LocalAsrTests(GenieTestCase):
    def test_streaming_recognition_of_a_known_phrase(self):
        from interviewgenie.asr.local import LocalStreamingASR

        asr = LocalStreamingASR()
        phrase = "tell me about yourself"
        audio = synth_audio(phrase, asr.sample_rate)
        asr.reset()
        chunk = int(asr.sample_rate * 0.32)
        for offset in range(0, len(audio.samples), chunk):
            asr.accept_audio(audio.samples[offset:offset + chunk], asr.sample_rate)
        asr.flush()
        chunks = asr.poll()
        final = next((c for c in reversed(chunks) if c.is_final), None)
        self.assertIsNotNone(final)
        self.assertEqual(final.text, phrase)
        self.assertGreater(final.confidence, 0.0)

    def test_partial_results_are_emitted_before_the_final(self):
        from interviewgenie.asr.local import LocalStreamingASR

        asr = LocalStreamingASR(partial_interval_ms=100)
        audio = synth_audio("what is your greatest weakness", asr.sample_rate)
        asr.reset()
        chunk = int(asr.sample_rate * 0.32)
        for offset in range(0, len(audio.samples), chunk):
            asr.accept_audio(audio.samples[offset:offset + chunk], asr.sample_rate)
        asr.flush()
        chunks = asr.poll()
        self.assertTrue(any(not c.is_final for c in chunks), "expected interim hypotheses")
        self.assertTrue(any(c.is_final for c in chunks))

    def test_confidence_drops_in_noise(self):
        from interviewgenie.asr.denoise import add_noise
        from interviewgenie.asr.local import LocalStreamingASR

        asr = LocalStreamingASR()
        phrase = "tell me about yourself"
        audio = synth_audio(phrase, asr.sample_rate)
        chunk = int(asr.sample_rate * 0.32)

        def run(samples):
            asr.reset()
            for offset in range(0, len(samples), chunk):
                asr.accept_audio(samples[offset:offset + chunk], asr.sample_rate)
            asr.flush()
            final = next((c for c in reversed(asr.poll()) if c.is_final), None)
            return final.confidence if final else 0.0

        clean_confidence = run(audio.samples)
        noisy = add_noise(audio, 0, "babble", seed=3)
        noisy_confidence = run(noisy.samples)
        self.assertLessEqual(noisy_confidence, clean_confidence + 1e-9)

    def test_reset_clears_state(self):
        from interviewgenie.asr.local import LocalStreamingASR

        asr = LocalStreamingASR()
        asr.accept_audio(tone(300, 0.5), asr.sample_rate)
        asr.reset()
        self.assertEqual(asr.poll(), [])
        self.assertEqual(asr._audio, [])

    def test_calibration_adapts_the_bank(self):
        from interviewgenie.asr.local import LocalStreamingASR

        asr = LocalStreamingASR()
        asr.ensure_seeded()
        before = asr.bank.template_for("tell me about yourself").confidence
        asr.calibrate("tell me about yourself",
                      synth_audio("tell me about yourself", asr.sample_rate))
        after = asr.bank.template_for("tell me about yourself")
        self.assertNotEqual(before, after.confidence)


class ProviderTests(GenieTestCase):
    def test_mock_provider_replays_a_script(self):
        from interviewgenie.asr.mock import MockASRProvider
        from interviewgenie.types import TranscriptChunk

        provider = MockASRProvider(script=["first question", "second question"])
        provider.accept_audio([0.0] * 16000, 16000)
        chunks = provider.poll()
        self.assertTrue(chunks)
        self.assertTrue(all(isinstance(c, TranscriptChunk) for c in chunks))

    def test_mock_provider_applies_an_error_model(self):
        from interviewgenie.asr.mock import MockASRProvider

        provider = MockASRProvider(script=["the quick brown fox"], word_error_rate=1.0,
                                   seed=1)
        provider.accept_audio([0.0] * 16000, 16000)
        final = next(c for c in provider.poll() if c.is_final)
        self.assertNotEqual(final.text, "the quick brown fox")

    def test_provider_factory_selects_the_local_engine(self):
        from interviewgenie.asr.base import provider_from_config

        provider = provider_from_config({"asr": {"provider": "local"}})
        self.assertEqual(provider.name, "local")

    def test_provider_factory_rejects_unknown_providers(self):
        from interviewgenie.asr.base import provider_from_config
        from interviewgenie.errors import ASRError

        with self.assertRaises(ASRError):
            provider_from_config({"asr": {"provider": "nope"}})

    def test_cloud_provider_falls_back_when_credentials_are_missing(self):
        from interviewgenie.asr.cloud import CloudASRProvider
        from interviewgenie.asr.local import LocalStreamingASR

        provider = CloudASRProvider(vendor="google", credentials={})
        provider.accept_audio([0.0] * 16000, 16000)
        # no credentials -> the cloud call fails, the local engine takes over
        provider.flush()
        self.assertIsInstance(provider._fallback, LocalStreamingASR)
        self.assertEqual(provider.capabilities().streaming, True)

    def test_websocket_accept_key_matches_the_rfc_example(self):
        from interviewgenie.api.ws import accept_key

        self.assertEqual(
            accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")

    def test_noise_report_is_reproducible(self):
        """Regression: the synthesiser seeded its excitation with ``hash()``,
        which Python randomises per process, so every run disagreed."""
        from interviewgenie.cli import noise_report

        first = noise_report()
        second = noise_report()
        self.assertEqual(len(first), 16)
        self.assertEqual(first, second)
        self.assertEqual([r["noise"] for r in first][:4], ["white"] * 4)
        self.assertEqual([r["snr"] for r in first][:4], [20, 10, 5, 0])

    def test_noise_suppression_actually_helps(self):
        from interviewgenie.cli import noise_report

        rows = noise_report()
        self.assertTrue(all(r["gain"] > 0 for r in rows), msg=str(rows))
        # Stationary noise (hum) is suppressed far more than speech-shaped babble.
        hum = max(r["gain"] for r in rows if r["noise"] == "hum")
        babble = min(r["gain"] for r in rows if r["noise"] == "babble")
        self.assertGreater(hum, babble)

    def test_websocket_read_exact_survives_a_short_read(self):
        """Regression: the reader caught a nonexistent
        ``asyncio.IncompleteCompleteError``, so every client disconnect raised."""
        import asyncio

        from interviewgenie.api.ws import WebSocketConnection

        async def go():
            # Built inside the coroutine: StreamReader() needs a running loop.
            reader = asyncio.StreamReader()
            reader.feed_data(b"\x81\x05ab")      # declares 5 bytes, sends 2
            reader.feed_eof()
            conn = WebSocketConnection(reader=reader, writer=None)
            return await conn._read_exact(5)

        self.assertIsNone(asyncio.run(go()))

    def test_websocket_accept_key_matches_the_rfc_example(self):
        from interviewgenie.api.ws import accept_key

        self.assertEqual(
            accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
