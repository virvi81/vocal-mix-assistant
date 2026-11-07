"""Regression checks using numeric fixtures, without audio files or playback."""
import unittest
from unittest.mock import patch
import numpy as np
from core import Clip, RATE
from proposals import suggest, render


class ProposalsTest(unittest.TestCase):
    def setUp(self):
        samples = np.full((15 * RATE, 2), .1, dtype=np.float32)
        self.voice = Clip(samples, "", RATE, 2, 15, 45, 15)
        self.music = Clip(samples.copy(), "", RATE, 2, 15, 45, 15)

    def measurements(self, values, target=-2):
        return {"baseline_db": target, "rows": [
            {"start": 45 + 5*i, "end": 50 + 5*i,
             "ratio_db": value, "comparable_pct": 100}
            for i, value in enumerate(values)]}

    def test_rebound_does_not_trigger_opposite_compensation(self):
        with patch("proposals.measure", return_value=self.measurements([-2, -6, -2])):
            proposals = suggest(self.voice, self.music)
        self.assertEqual([(p["start"], p["gain_db"]) for p in proposals], [(50, 1)])

    def test_accepted_gain_is_measured_and_locked_target_stays_original(self):
        accepted = [{"start": 50, "end": 55, "gain_db": -1}]
        original = self.voice.samples.copy()
        with patch("proposals.measure", side_effect=[
                self.measurements([-2, 1, -2]),
                self.measurements([-2, 0, -4], target=-3)]) as measurement:
            proposals = suggest(self.voice, self.music, accepted)
        adjusted = measurement.call_args_list[1].args[0].samples
        self.assertLess(adjusted[7 * RATE, 0], original[7 * RATE, 0])
        self.assertEqual([(p["start"], p["gain_db"]) for p in proposals], [(55, 1)])
        self.assertEqual(proposals[0]["target_db"], -2)
        np.testing.assert_array_equal(self.voice.samples, original)

    def test_undo_releases_interval(self):
        with patch("proposals.measure", return_value=self.measurements([-2, 1, -2])):
            self.assertEqual(suggest(self.voice, self.music)[0]["start"], 50)

    def test_reject_overlapping_acceptances(self):
        with self.assertRaises(ValueError):
            render(self.voice.samples, self.music.samples, 45, [
                {"start": 45, "end": 52, "gain_db": 1},
                {"start": 50, "end": 55, "gain_db": -1}])


if __name__ == "__main__":
    unittest.main()
