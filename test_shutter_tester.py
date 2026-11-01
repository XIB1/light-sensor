"""Regression checks for the v12 merge; run with python -m unittest."""
import unittest
from types import SimpleNamespace

import numpy as np

import shutter_tester as tester


class MergeRegressionTests(unittest.TestCase):
    def signal(self, speed):
        return tester.simulate_shutter(
            speed, 3.3, 3.3, 24.0, 22.0, 2.0,
            rng=np.random.default_rng(42),
        )

    def analyze(self, samples, speed, spacing=22.0):
        return tester.analyze_signal(samples, speed, 3.3, 24.0, spacing, 2.0)

    def test_reversed_and_mixed_polarity(self):
        for speed in (1 / 30, 1 / 4000):
            samples = self.signal(speed)
            reference = self.analyze(samples, speed)
            for signs in ((-1, -1), (-1, 1)):
                with self.subTest(speed=speed, signs=signs):
                    result = self.analyze(samples * signs, speed)
                    self.assertAlmostEqual(result.exposure_average_s, reference.exposure_average_s)
                    self.assertEqual(result.event_indices, reference.event_indices)

    def test_spacing_scales_travel_only(self):
        samples = self.signal(1 / 30)
        full = self.analyze(samples, 1 / 30)
        half = self.analyze(samples, 1 / 30, spacing=11.0)
        self.assertEqual(full.exposure_average_s, half.exposure_average_s)
        self.assertAlmostEqual(half.first_curtain_full_travel_s,
                               full.first_curtain_full_travel_s * 2)

    def test_live_opening_after_long_idle(self):
        speed = 1 / 30
        samples = -self.signal(speed)
        delayed = np.concatenate((np.zeros((tester.FS, 2)), samples))
        recorded = []
        variable = lambda value: SimpleNamespace(get=lambda: value)
        app = SimpleNamespace(
            nominal_var=variable("1/30"), coupling_var=variable("AC coupled"),
            live_peak_floor=0.025, _parse_geometry=lambda: (24, 22, 2, 3.3),
            _update_results=lambda *args: None, _update_plot=lambda *args: None,
            _record_summary=lambda result, nominal: recorded.append(result),
            status_var=SimpleNamespace(set=lambda text: None),
        )
        tester.ShutterTesterApp._analyze_and_display(app, delayed, "live audio")
        self.assertEqual(len(recorded), 1)
        self.assertLess(abs(recorded[0].exposure_average_s - speed), 0.001)
        self.assertLess(recorded[0].event_indices["sensor1_open"], int(0.040 * tester.FS))


if __name__ == "__main__":
    unittest.main()
