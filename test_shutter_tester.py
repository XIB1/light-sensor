"""Regression checks for the v12 merge; run with python -m unittest."""
import unittest
import json
import tempfile
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
            ac_method_var=variable("Auto"),
            debug_mode_var=variable(False),
            live_peak_floor=0.025, _parse_geometry=lambda: (24, 22, 2, 3.3),
            _update_results=lambda *args: None, _update_plot=lambda *args: None,
            _record_summary=lambda result, nominal: recorded.append(result),
            status_var=SimpleNamespace(set=lambda text: None),
        )
        tester.ShutterTesterApp._analyze_and_display(app, delayed, "live audio")
        self.assertEqual(len(recorded), 1)
        self.assertLess(abs(recorded[0].exposure_average_s - speed), 0.001)
        self.assertLess(recorded[0].event_indices["sensor1_open"], int(0.040 * tester.FS))

    def test_fast_speeds_use_pulse_fit_and_recover_exposure(self):
        for speed in (1 / 1000, 1 / 2000, 1 / 4000, 1 / 8000):
            with self.subTest(speed=speed):
                result = self.analyze(-self.signal(speed), speed)
                self.assertEqual(result.method, "AC template fit")
                self.assertLess(abs(result.exposure_average_s / speed - 1), 0.1)
                for sensor in (1, 2):
                    self.assertLess(result.event_indices[f"sensor{sensor}_open"],
                                    result.event_indices[f"sensor{sensor}_close"])

    def test_auto_uses_observed_duration_near_transition(self):
        samples = self.signal(1 / 1000)
        result = self.analyze(samples, 1 / 500)
        self.assertEqual(result.method, "AC template fit")
        self.assertLess(abs(result.exposure_average_s * 1000 - 1), 0.1)

    def test_manual_ac_method_override(self):
        result = tester.analyze_signal(
            self.signal(1 / 500), 1 / 500, 3.3, 24, 22, 2,
            ac_method="AC pulse fit",
        )
        self.assertEqual(result.method, "AC template fit")
        result = tester.analyze_signal(
            self.signal(1 / 1000), 1 / 1000, 3.3, 24, 22, 2,
            ac_method="AC transient edges",
        )
        self.assertEqual(result.method, "AC transient edges")

    def test_plot_zooms_to_event_and_rescales_after_slow_test(self):
        figure = tester.Figure()
        app = SimpleNamespace(
            ax=figure.add_subplot(111), fig=figure,
            canvas=SimpleNamespace(draw_idle=lambda: None),
        )
        widths = []
        for speed in (1 / 30, 1 / 4000):
            samples = self.signal(speed)
            result = self.analyze(samples, speed)
            # Simulate extra live audio after the event.
            samples = np.concatenate((samples, np.zeros((tester.FS, 2))))
            tester.ShutterTesterApp._update_plot(app, samples, result)
            left, right = app.ax.get_xlim()
            widths.append(right - left)
            for index in result.event_indices.values():
                self.assertLess(left, index / tester.FS * 1000)
                self.assertGreater(right, index / tester.FS * 1000)
        self.assertLess(widths[1], widths[0] / 4)
        self.assertLess(widths[1], 10)
        labels = app.ax.get_legend_handles_labels()[1]
        self.assertIn("S1 open (fit)", labels)
        self.assertIn("S2 close (fit)", labels)


class DebugAndSummaryTests(unittest.TestCase):
    def make_app(self, directory, debug=True):
        variable = lambda value: SimpleNamespace(get=lambda: value, set=lambda text: None)
        messages = []
        app = SimpleNamespace(
            nominal_var=variable("1/1000"), source_var=variable("Live test input"),
            camera_var=variable("Nikon FE2"), path_var=variable("24"),
            spacing_var=variable("21"), aperture_var=variable("2"),
            expected_travel_var=variable("3.3"), coupling_var=variable("AC coupled"),
            ac_method_var=variable("Auto"), tolerance_var=variable("±1/3 stop"),
            travel_tolerance_var=variable("Auto (service / ±5%)"),
            mismatch_tolerance_var=variable("Auto (service / ±5%)"),
            debug_mode_var=variable(debug), debug_directory=directory,
            debug_status_var=SimpleNamespace(set=messages.append),
            live_peak_floor=0.025, _parse_geometry=lambda: (24, 21, 2, 3.3),
            _update_results=lambda *args: None, _update_plot=lambda *args: None,
            _record_summary=lambda *args: None, status_var=variable(""),
            pass_fail_var=variable("PASS"), tolerance_detail_var=variable("Test assessment"),
        )
        app._debug_settings = lambda source: tester.ShutterTesterApp._debug_settings(app, source)
        return app, messages

    def signal(self):
        signal = -tester.simulate_shutter(
            1 / 1000, 3.3, 3.3, 24, 21, 2, rng=np.random.default_rng(42)
        )
        return np.concatenate((np.zeros((tester.FS, 2), dtype=np.float32), signal))

    def test_debug_capture_preserves_samples_settings_and_replays_result(self):
        with tempfile.TemporaryDirectory() as directory:
            app, messages = self.make_app(directory)
            raw = self.signal()
            tester.ShutterTesterApp._analyze_and_display(app, raw, "live audio")
            archives = list(Path(directory).glob("*.npz"))
            self.assertEqual(len(archives), 1, messages)
            report = json.loads(archives[0].with_suffix(".json").read_text(encoding="utf-8"))
            with np.load(archives[0], allow_pickle=False) as capture:
                self.assertEqual(json.loads(str(capture["metadata_json"])), report)
                np.testing.assert_array_equal(capture["raw_samples"], raw)
                np.testing.assert_array_equal(capture["samples"], raw[report["analysis_start_frame"]:])
                settings = report["settings"]
                self.assertEqual(settings["sensor_spacing_mm"], 21)
                self.assertEqual(settings["optical_aperture_mm"], 2)
                self.assertEqual(settings["sample_rate_hz"], tester.FS)
                replay = tester.analyze_signal(
                    capture["samples"], settings["nominal_s"], settings["expected_travel_ms"],
                    settings["reference_distance_mm"], settings["sensor_spacing_mm"],
                    settings["optical_aperture_mm"], settings["coupling"],
                    fs=settings["sample_rate_hz"], ac_method=settings["ac_method"],
                )
            self.assertEqual(replay.exposure_average_s, report["result"]["exposure_average_s"])
            self.assertEqual(replay.event_indices, report["result"]["event_indices"])
            self.assertEqual(report["outcome"], "completed")
            order = tester.infer_sensor_channels(replay.event_indices)
            self.assertEqual(report["sensor_order"]["first_input_channel"], order[0])
            self.assertEqual(report["sensor_order"]["second_input_channel"], order[1])
            self.assertEqual(report["sensor_order"]["result_field_numbering"], "physical input channel")
            for name, index in replay.event_indices.items():
                self.assertEqual(report["raw_event_indices"][name], index + report["analysis_start_frame"])

    def test_rejected_capture_is_saved_with_error(self):
        with tempfile.TemporaryDirectory() as directory:
            app, messages = self.make_app(directory)
            with self.assertRaisesRegex(ValueError, "No clear shutter opening"):
                tester.ShutterTesterApp._analyze_and_display(
                    app, np.zeros((tester.FS, 2), dtype=np.float32), "live audio"
                )
            reports = list(Path(directory).glob("*.json"))
            self.assertEqual(len(reports), 1, messages)
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(report["outcome"], "rejected")
            self.assertIn("No clear shutter opening", report["error"])
            self.assertIsNone(report["result"])

    def test_debug_disabled_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            app, _ = self.make_app(directory, debug=False)
            tester.ShutterTesterApp._analyze_and_display(app, self.signal(), "live audio")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_debug_write_failure_keeps_measurement(self):
        app, messages = self.make_app("unused")
        with patch.object(tester, "write_debug_capture", side_effect=OSError("disk full")):
            tester.ShutterTesterApp._analyze_and_display(app, self.signal(), "live audio")
        self.assertGreater(app.result.exposure_average_s, 0)
        self.assertIn("Debug save failed: disk full", messages)

    def test_summary_formats_signed_stops_from_exposure_duration(self):
        self.assertEqual(tester.format_summary_measurement(0.002, 0.001), "2.000 (+1.00 stops)")
        self.assertEqual(tester.format_summary_measurement(0.0005, 0.001), "0.500 (-1.00 stops)")
        self.assertEqual(tester.format_summary_measurement(0.001, 0.001), "1.000 (+0.00 stops)")
        rows = []
        app = SimpleNamespace(
            summary_count_var=SimpleNamespace(get=lambda: "2"),
            source_var=SimpleNamespace(get=lambda: "Live"),
            summary_source_var=SimpleNamespace(set=lambda text: None),
            test_history={"Live": {0.001: deque([0.01, 0.0005, 0.002], maxlen=20)}},
            summary_table=SimpleNamespace(
                get_children=lambda: (), delete=lambda *args: None,
                insert=lambda *args, **kwargs: rows.append(kwargs["values"]),
            ),
        )
        tester.ShutterTesterApp._refresh_summary(app)
        row = next(row for row in rows if row[0] == "1/1000")
        self.assertEqual(row, ("1/1000", "2/2", "1.250 (+0.32 stops)",
                               "0.500 (-1.00 stops)", "2.000 (+1.00 stops)"))


class SensorOrderTests(unittest.TestCase):
    def result(self, reverse=False, simultaneous=False):
        opens = [1500, 1000]
        durations = [864, 672]
        if reverse:
            opens.reverse()
            durations.reverse()
        if simultaneous:
            opens = [1000, 1000]
        closes = [o + d for o, d in zip(opens, durations)]
        delays = [(opens[1] - opens[0]) / tester.FS,
                  (closes[1] - closes[0]) / tester.FS]
        return tester.AnalysisResult(
            method="AC transient edges", exposure_sensor_1_s=durations[0] / tester.FS,
            exposure_sensor_2_s=durations[1] / tester.FS, exposure_average_s=0.008,
            first_sensor_delay_s=delays[0], second_sensor_delay_s=delays[1],
            first_curtain_full_travel_s=abs(delays[0]) * 24 / 21,
            second_curtain_full_travel_s=abs(delays[1]) * 24 / 21,
            sensor_mismatch_percent=25, confidence=1.0,
            event_indices={"sensor1_open": opens[0], "sensor2_open": opens[1],
                           "sensor1_close": closes[0], "sensor2_close": closes[1]},
        )

    def app(self):
        class Variable:
            def __init__(self, value=""):
                self.value = value
            def get(self):
                return self.value
            def set(self, value):
                self.value = value
        app = tester.ShutterTesterApp.__new__(tester.ShutterTesterApp)
        for name in (
            "method", "avg", "s1", "s2", "variance", "s1_variance", "s2_variance",
            "s1_channel", "s2_channel", "s1_duration", "s2_duration", "avg_duration",
            "avg_variance", "first_sensor_title", "second_sensor_title", "sensor_order",
            "mismatch", "conf", "travel1", "travel2", "curtain_mismatch",
            "travel1_status", "travel2_status", "curtain_mismatch_status", "tolerance_detail",
        ):
            setattr(app, name + "_var", Variable())
        app.tolerance_var = Variable("±1/3 stop")
        app.travel_tolerance_var = Variable("±5%")
        app.mismatch_tolerance_var = Variable("±5%")
        app.camera_var = Variable("Manual")
        app.exposure_status_vars = {name: Variable("NOT TESTED") for name in ("first", "average", "second")}
        app.exposure_status_labels = {
            name: SimpleNamespace(configure=lambda **kwargs: None)
            for name in ("first", "average", "second")
        }
        for name in ("t1_dot", "t1_circle", "t1_label", "t2_dot", "t2_circle", "t2_label",
                     "tm_dot", "tm_circle", "tm_label"):
            setattr(app, name, None)
        app._set_indicator = lambda passed: None
        app._set_small_indicator = lambda *args: None
        return app

    def test_earliest_opening_infers_order_and_handles_unknown(self):
        self.assertEqual(tester.infer_sensor_channels(self.result().event_indices), (2, 1))
        self.assertEqual(tester.infer_sensor_channels(self.result(reverse=True).event_indices), (1, 2))
        self.assertIsNone(tester.infer_sensor_channels(self.result(simultaneous=True).event_indices))
        self.assertIsNone(tester.infer_sensor_channels({}))

    def test_swapping_inputs_preserves_first_average_second_readings(self):
        app = self.app()
        for reverse, channel in ((False, 2), (True, 1)):
            with self.subTest(reverse=reverse):
                result = self.result(reverse=reverse)
                original_indices = result.event_indices.copy()
                app._update_results(result, 0.008, 24, 6)
                self.assertEqual(app.s1_channel_var.get(), f"Input channel {channel}")
                self.assertEqual(app.s1_duration_var.get(), "7.0000 ms")
                self.assertEqual(app.avg_duration_var.get(), "8.0000 ms")
                self.assertEqual(app.s2_duration_var.get(), "9.0000 ms")
                self.assertEqual(app.s1_variance_var.get(), "-0.19 stops")
                self.assertEqual(app.s2_variance_var.get(), "+0.17 stops")
                self.assertEqual(result.event_indices, original_indices)

    def test_simultaneous_opening_is_not_labelled_as_known_order(self):
        app = self.app()
        app._update_results(self.result(simultaneous=True), 0.008, 24, 6)
        self.assertEqual(app.first_sensor_title_var.get(), "Sensor A")
        self.assertEqual(app.second_sensor_title_var.get(), "Sensor B")
        self.assertIn("unresolved", app.sensor_order_var.get())

    def test_individual_exposure_verdicts_cannot_be_hidden_by_average(self):
        app = self.app()
        result = self.result()
        # Physical channel 2 is reached first and is on target. Channel 1,
        # reached second, exceeds 1/3 stop despite the average passing.
        result.exposure_sensor_1_s = 0.011
        result.exposure_sensor_2_s = 0.008
        result.exposure_average_s = 0.0095
        result.first_curtain_full_travel_s = 0.006
        result.second_curtain_full_travel_s = 0.006
        overall = []
        app._set_indicator = overall.append
        colors = {}
        for name in app.exposure_status_labels:
            app.exposure_status_labels[name].configure = lambda name=name, **kwargs: colors.update({name: kwargs["foreground"]})
        app._update_results(result, 0.008, 24, 6)
        self.assertEqual({name: var.get() for name, var in app.exposure_status_vars.items()},
                         {"first": "PASS", "average": "PASS", "second": "FAIL"})
        self.assertEqual(colors["first"], "#20743b")
        self.assertEqual(colors["second"], "#a32727")
        self.assertFalse(overall[-1])
        self.assertIn("Exposure fails", app.tolerance_detail_var.get())

        app.result = result
        app.result_context = (0.008, 24, 6)
        app.tolerance_var.set("±1/2 stop")
        app._refresh_result_assessment()
        self.assertTrue(all(var.get() == "PASS" for var in app.exposure_status_vars.values()))
        self.assertTrue(overall[-1])

    def test_plot_trace_and_markers_follow_inferred_sensor(self):
        figure = tester.Figure()
        app = SimpleNamespace(ax=figure.add_subplot(111), fig=figure,
                              canvas=SimpleNamespace(draw_idle=lambda: None))
        samples = np.column_stack((np.arange(3000), -np.arange(3000)))
        result = self.result()
        tester.ShutterTesterApp._update_plot(app, samples, result)
        first_trace = app.ax.lines[0]
        self.assertEqual(first_trace.get_label(), "First sensor (input channel 2)")
        lo, hi = tester.waveform_view_bounds(result.event_indices, len(samples))
        np.testing.assert_array_equal(first_trace.get_ydata(), samples[lo:hi, 1])
        marker = next(line for line in app.ax.lines if line.get_label() == "S1 open")
        self.assertAlmostEqual(marker.get_xdata()[0], result.event_indices["sensor2_open"] / tester.FS * 1000)


class F80PresetTests(unittest.TestCase):
    def test_documented_target_uses_21mm_reference_and_service_range(self):
        preset = tester.CAMERA_BY_NAME["Nikon F80 / N80"]
        self.assertEqual(preset.path_mm, 21)
        self.assertEqual(preset.travel_ms, 5.25)
        self.assertEqual(preset.travel_quality, "factory")
        self.assertEqual(preset.service_travel_min_ms, 5.0)
        self.assertEqual(preset.service_travel_max_ms, 5.5)

    def test_simulator_preserves_21mm_spacing_at_21mm_reference(self):
        samples = tester.simulate_shutter(
            0.008, 5.25, 5.25, 21, 21, 2, rng=np.random.default_rng(42)
        )
        result = tester.analyze_signal(samples, 0.008, 5.25, 21, 21, 2)
        self.assertAlmostEqual(abs(result.first_sensor_delay_s), 0.00525, delta=2 / tester.FS)
        self.assertAlmostEqual(result.first_curtain_full_travel_s, 0.00525, delta=2 / tester.FS)
        self.assertAlmostEqual(result.second_curtain_full_travel_s, 0.00525, delta=2 / tester.FS)

    def test_auto_travel_assessment_uses_documented_limits(self):
        app = SensorOrderTests().app()
        app.camera_var.set("Nikon F80 / N80")
        app.travel_tolerance_var.set("Auto (service / ±5%)")
        indicators = []
        app._set_small_indicator = lambda *args: indicators.append(bool(args[-1]))
        result = SensorOrderTests().result()
        for travel_ms, passed in ((5.052083, True), (5.0, True), (5.5, True), (5.6, False)):
            with self.subTest(travel_ms=travel_ms):
                indicators.clear()
                result.first_curtain_full_travel_s = travel_ms / 1000
                result.second_curtain_full_travel_s = travel_ms / 1000
                app._update_results(result, 0.008, 21, 5.25)
                self.assertEqual(indicators[:2], [passed, passed])
                self.assertIn("service 5–5.5 ms", app.tolerance_detail_var.get())


if __name__ == "__main__":
    unittest.main()
