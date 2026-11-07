"""Numeric unit fixtures only: no song files, model inference, playback or audio writes."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from core import Clip, RATE, load_clip, measure, playback_sources, audition_slice


def clip(data, start=45):
    return Clip(data.astype("float32"), "numeric-test-fixture", RATE, 2,
                len(data)/RATE, start, len(data)/RATE)


def constant(value, seconds=5):
    return np.full((round(seconds*RATE), 2), value, dtype="float32")


class BalanceTests(unittest.TestCase):
    def test_known_amplitude_ratio_not_loudness(self):
        result = measure(clip(constant(.1)), clip(constant(.2)))
        self.assertAlmostEqual(result["baseline_db"], -6.020599913, places=5)
        self.assertFalse(result["rows"][0]["review_change"])

    def test_antiphase_stereo_is_not_cancelled_before_measurement(self):
        data = constant(.1); data[:, 1] *= -1
        result = measure(clip(data), clip(constant(.2)))
        self.assertAlmostEqual(result["baseline_db"], -6.020599913, places=5)

    def test_silence_produces_no_comparison_not_zero_db(self):
        result = measure(clip(constant(0)), clip(constant(.2)))
        self.assertIsNone(result["baseline_db"])
        self.assertIsNone(result["rows"][0]["ratio_db"])
        self.assertFalse(result["rows"][0]["review_change"])

    def test_missing_accompaniment_blocks_comparison(self):
        result = measure(clip(constant(.1)), clip(constant(0)))
        self.assertIsNone(result["baseline_db"])
        self.assertEqual(result["rows"][0]["instrumental_active_pct"], 0)

    def test_change_is_located_in_original_time(self):
        voice = np.concatenate([constant(.1, 10), constant(.05, 5), constant(.1, 5)])
        result = measure(clip(voice), clip(constant(.2, 20)))
        flagged = [r for r in result["rows"] if r["review_change"]]
        self.assertEqual(len(flagged), 1)
        self.assertEqual((flagged[0]["start"], flagged[0]["end"]), (55, 60))
        self.assertLess(flagged[0]["difference_db"], -5.9)

    def test_nonmatching_shapes_or_time_are_blocked(self):
        with self.assertRaises(ValueError):
            measure(clip(constant(.1)), clip(constant(.2, 4)))
        with self.assertRaises(ValueError):
            measure(clip(constant(.1), 45), clip(constant(.2), 46))

    def test_common_attenuation_preserves_ab_amplitude_difference(self):
        data = np.concatenate([constant(.8, 5), constant(.4, 5)])
        voice, music = clip(data), clip(constant(.8, 10))
        sources, gain = playback_sources(voice, music)
        self.assertLess(gain, 1)
        a = audition_slice(sources["Solo voz"], 45, 45, 50, gain)
        b = audition_slice(sources["Solo voz"], 45, 50, 55, gain)
        self.assertAlmostEqual(float(a.max()/b.max()), 2)
        self.assertLessEqual(float(np.max(np.abs(sources["Suma de stems"]*gain))), .890001)
        with self.assertRaises(ValueError):
            audition_slice(sources["Solo voz"], 45, 40, 46, gain)

    def test_decode_seeks_only_requested_interval_without_writing(self):
        fake = MagicMock()
        fake.samplerate, fake.channels = RATE, 1
        fake.__len__.return_value = 10 * RATE
        fake.read.return_value = np.full((2*RATE, 1), .1, dtype="float32")
        manager = MagicMock()
        manager.__enter__.return_value = fake
        info = SimpleNamespace(st_size=123, st_mtime_ns=456)
        with patch.object(Path, "is_file", return_value=True), patch.object(Path, "stat", return_value=info), patch("core.sf.SoundFile", return_value=manager) as opened:
            result = load_clip("/nonexistent-numeric-fixture", 3, 2)
        opened.assert_called_once_with("/nonexistent-numeric-fixture")
        fake.seek.assert_called_once_with(3*RATE)
        fake.read.assert_called_once_with(2*RATE, dtype="float32", always_2d=True)
        self.assertEqual(result.samples.shape, (2*RATE, 2))
        self.assertEqual(result.source_channels, 1)

    def test_no_silent_truncation_to_shared_length(self):
        fake = MagicMock()
        fake.samplerate, fake.channels = RATE, 2
        fake.__len__.return_value = 2*RATE
        manager = MagicMock(); manager.__enter__.return_value = fake
        info = SimpleNamespace(st_size=123, st_mtime_ns=456)
        with patch.object(Path, "is_file", return_value=True), patch.object(Path, "stat", return_value=info), patch("core.sf.SoundFile", return_value=manager):
            with self.assertRaises(ValueError): load_clip("/nonexistent-numeric-fixture", 1, 2)
        fake.read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
