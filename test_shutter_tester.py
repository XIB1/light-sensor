"""Regression checks for the v12 merge; run with python -m unittest."""
import unittest
import json
import tempfile
import threading
from collections import deque
from dataclasses import asdict, replace
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
        return tester.analyze_signal(samples, speed, 3.3, 24.0, spacing, 2.0, ac_method="Auto")

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

    def test_slow_live_capture_finishes_after_both_closes_and_tail(self):
        fs = tester.FS
        for marked, duration, coupling in (
            ("1", 1.0, "AC coupled"),
            ("1/2", 0.5, "DC / direct coupled"),
            ("1/2", 0.15, "AC coupled"),
        ):
            with self.subTest(marked=marked, duration=duration, coupling=coupling):
                app, _ = self.make_app("unused", debug=False)
                app.nominal_var = SimpleNamespace(get=lambda: marked)
                app.coupling_var = SimpleNamespace(get=lambda: coupling)
                app.ac_method_var = SimpleNamespace(get=lambda: tester.DEFAULT_AC_METHOD)
                app.reference_pending = False
                app.live_chunks = deque()
                app.live_samples = 0
                app.live_lock = threading.Lock()
                app.live_trigger_time = None
                app.live_cooldown_until = 0
                results = []
                app._record_summary = lambda result, nominal: results.append(result)
                app._analyze_and_display = lambda *args, **kwargs: (
                    tester.ShutterTesterApp._analyze_and_display(app, *args, **kwargs)
                )
                samples = tester.simulate_shutter(
                    duration, 3.3, 3.3, 24, 21, 2, coupling, rng=np.random.default_rng(42)
                ) * [-1, 1]
                cursor = 0

                def poll(until):
                    nonlocal cursor
                    end = round(until * fs)
                    tester.ShutterTesterApp._audio_callback(app, samples[cursor:end], end - cursor, None, None)
                    cursor = end
                    with patch.object(tester.time, "monotonic", return_value=until):
                        tester.ShutterTesterApp._process_live_state(app)

                poll(0.08)
                self.assertIsNotNone(app.live_trigger_time)
                poll(duration + 0.019)
                self.assertEqual(results, [])
                poll(duration + 0.022)
                self.assertEqual(results, [])
                poll(duration + 0.044)
                self.assertEqual(len(results), 1)
                self.assertAlmostEqual(results[0].exposure_average_s, duration, delta=0.001)
                self.assertIsNone(app.live_trigger_time)

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
            test_history={0.001: deque([0.01, 0.0005, 0.002], maxlen=20)},
            summary_table=SimpleNamespace(
                get_children=lambda: (), delete=lambda *args: None,
                insert=lambda *args, **kwargs: rows.append(kwargs["values"]),
            ),
            _update_speed_checkmarks=lambda: None,
        )
        tester.ShutterTesterApp._refresh_summary(app)
        self.assertEqual(len(rows), 1)
        row = next(row for row in rows if row[0] == "1/1000")
        self.assertEqual(row, ("1/1000", "2/2", "0.500 (-1.00 stops)",
                               "1.250 (+0.32 stops)", "2.000 (+1.00 stops)"))
        rows.clear()
        app.test_history[1/60] = deque([1/60], maxlen=20)
        app.test_history[1/8000] = deque(maxlen=20)
        tester.ShutterTesterApp._refresh_summary(app)
        self.assertEqual([row[0] for row in rows], ["1/60", "1/1000"])
        self.assertEqual(rows[0][1], "1/2")
        rows.clear()
        app.test_history.clear()
        tester.ShutterTesterApp._refresh_summary(app)
        self.assertEqual(rows, [])


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


class TransitionAndCompletionTests(unittest.TestCase):
    def test_default_method_remains_transient_edges_at_fast_speeds(self):
        samples, _ = self.pulse(exposure=1 / 4000)
        result = tester.analyze_signal(samples, 1 / 4000, 3.3, 24, 21, 2)
        self.assertEqual(tester.DEFAULT_AC_METHOD, "AC transient edges")
        self.assertEqual(result.method, "AC transient edges")

    def pulse(self, exposure=1 / 60, cutoff=2.5, closing_gain=1.0):
        fs = tester.FS
        crossing = 2 * 0.0033 / 24
        aperture_samples = int(round(crossing * fs))
        times = np.arange(int((0.020 + exposure + 0.080) * fs)) / fs
        channels = []
        centers = []
        for opened in (0.020, 0.020 + 21 * 0.0033 / 24):
            def response(edge):
                dt = times - edge
                return np.where(dt >= 0, np.exp(-2 * np.pi * cutoff * np.maximum(dt, 0)), 0.0)
            signal = -response(opened) + closing_gain * response(opened + exposure)
            signal = np.convolve(signal, np.ones(aperture_samples) / aperture_samples, mode="full")[:len(times)]
            channels.append(signal)
            centers.append((opened + (aperture_samples - 1) / (2 * fs),
                            opened + exposure + (aperture_samples - 1) / (2 * fs)))
        return np.column_stack(channels), centers

    def test_edges_mark_transition_midpoints_for_both_polarities(self):
        for cutoff in (2.5, 35.0):
            samples, centers = self.pulse(cutoff=cutoff)
            for polarity in (-1, 1):
                with self.subTest(cutoff=cutoff, polarity=polarity):
                    result = tester.analyze_signal(samples * polarity, 1 / 60, 3.3, 24, 21, 2,
                                                   ac_method="AC transient edges")
                    for sensor, (opened, closed) in enumerate(centers, 1):
                        self.assertAlmostEqual(result.event_indices[f"sensor{sensor}_open"] / tester.FS,
                                               opened, delta=2 / tester.FS)
                        self.assertAlmostEqual(result.event_indices[f"sensor{sensor}_close"] / tester.FS,
                                               closed, delta=2 / tester.FS)

    def test_long_pulse_fit_estimates_channel_decay_and_exposure(self):
        for exposure, cutoff in ((1 / 60, 2.5), (1 / 30, 35.0)):
            with self.subTest(exposure=exposure, cutoff=cutoff):
                samples, _ = self.pulse(exposure=exposure, cutoff=cutoff)
                result = tester.analyze_signal(samples, exposure, 3.3, 24, 21, 2,
                                               ac_method="AC pulse fit")
                self.assertLess(abs(result.exposure_average_s / exposure - 1), 0.03)
                self.assertIsNotNone(result.fitted_ac_cutoff_hz)
                for fitted in result.fitted_ac_cutoff_hz:
                    self.assertAlmostEqual(fitted, cutoff, delta=0.3)

    def test_stronger_closing_pulse_does_not_crop_out_opening(self):
        samples, _ = self.pulse(closing_gain=1.5, cutoff=35)
        result = tester.analyze_signal(samples, 1 / 60, 3.3, 24, 21, 2, ac_method="AC pulse fit")
        self.assertLess(abs(result.exposure_average_s * 60 - 1), 0.05)
        self.assertLess(result.event_indices["sensor1_open"] / tester.FS, 0.023)

    def test_fft_match_agrees_with_direct_correlation(self):
        rng = np.random.default_rng(42)
        template = rng.normal(size=800)
        signal = rng.normal(0, 0.01, 4000)
        signal[1234:2034] -= template
        score, start = tester._best_normalized_match(signal, template, allow_inverted=True)
        self.assertEqual(start, 1234)
        self.assertGreater(score, 0.999)

    def test_legend_is_outside_waveform_axes(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        figure = tester.Figure()
        canvas = FigureCanvasAgg(figure)
        app = SimpleNamespace(ax=figure.add_subplot(111), fig=figure,
                              canvas=SimpleNamespace(draw_idle=lambda: None))
        samples, _ = self.pulse()
        result = tester.analyze_signal(samples, 1 / 60, 3.3, 24, 21, 2)
        tester.ShutterTesterApp._update_plot(app, samples, result)
        canvas.draw()
        renderer = canvas.get_renderer()
        self.assertGreater(app.ax.get_legend().get_window_extent(renderer).x0,
                           app.ax.get_window_extent(renderer).x1)

    def test_speed_checkmarks_follow_window_and_survive_input_and_method_changes(self):
        self.assertFalse(tester.passing_test_window([True] * 4, 5))
        self.assertTrue(tester.passing_test_window([False] + [True] * 5, 5))
        self.assertFalse(tester.passing_test_window([True] * 4 + [False], 5))
        app = SensorOrderTests().app()
        variable = app.tolerance_var.__class__
        app.source_var = variable("Live A")
        app.summary_source_var = variable()
        app.summary_count_var = variable("2")
        app.pass_fail_var = variable("PASS")
        app.ac_method_var = variable(tester.DEFAULT_AC_METHOD)
        app._save_summary = lambda: None
        app._clear_live_state = lambda: None
        app.test_history, app.test_pass_history = {}, {}
        app.summary_table = SimpleNamespace(get_children=lambda: (), delete=lambda *args: None,
                                            insert=lambda *args, **kwargs: None)
        labels = {}
        app.speed_buttons = {"1/60": SimpleNamespace(configure=lambda **kwargs: labels.update(kwargs))}
        result = SensorOrderTests().result()
        app._record_summary(result, 1 / 60)
        self.assertEqual(labels["text"], "1/60")
        app._record_summary(result, 1 / 60)
        self.assertEqual(labels["text"], "✓ 1/60")
        app.pass_fail_var.set("FAIL")
        app._record_summary(result, 1 / 60)
        self.assertEqual(labels["text"], "1/60")
        app.pass_fail_var.set("PASS")
        app._record_summary(result, 1 / 60)
        self.assertEqual(labels["text"], "1/60")
        app.summary_count_var.set("1")
        app._refresh_summary()
        self.assertEqual(labels["text"], "✓ 1/60")
        app.source_var.set("Live B")
        app._refresh_summary()
        self.assertEqual(labels["text"], "✓ 1/60")
        app.ac_method_var.set(tester.INTEGRATION_METHOD)
        app._analysis_method_changed()
        self.assertEqual(labels["text"], "✓ 1/60")
        self.assertEqual(len(app.test_history[1/60]), 4)
        app._clear_summary()
        self.assertEqual(labels["text"], "1/60")
        self.assertEqual(app.test_pass_history, {})


class SummaryPersistenceTests(unittest.TestCase):
    def app(self, path):
        app = SensorOrderTests().app()
        variable = app.tolerance_var.__class__
        app.summary_path = path
        app.summary_save_blocked = False
        app.summary_count_var = variable("2")
        app.summary_storage_status_var = variable()
        app.source_var = variable("Live A")
        app.ac_method_var = variable(tester.DEFAULT_AC_METHOD)
        app.pass_fail_var = variable("PASS")
        app.test_history, app.test_pass_history = {}, {}
        app._refresh_summary = lambda: None
        return app

    def test_restart_restores_shared_measurements_verdicts_and_selected_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sheet.json"
            app = self.app(path)
            app._record_summary(SimpleNamespace(exposure_average_s=0.008), 1/125)
            app.source_var.set("Live B")
            app.ac_method_var.set(tester.INTEGRATION_METHOD)
            app.pass_fail_var.set("FAIL")
            app._record_summary(SimpleNamespace(exposure_average_s=0.007), 1/125)
            app.summary_count_var.set("7")
            app._summary_count_changed()
            restarted = self.app(path)
            restarted._restore_summary()
            self.assertEqual(list(restarted.test_history[1/125]), [0.008, 0.007])
            self.assertEqual(list(restarted.test_pass_history[1/125]), [True, False])
            self.assertEqual(restarted.summary_count_var.get(), "7")
            self.assertFalse(restarted.summary_save_blocked)

    def test_only_latest_twenty_per_speed_are_retained_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sheet.json"
            app = self.app(path)
            for index in range(25):
                app.pass_fail_var.set("PASS" if index % 2 else "FAIL")
                app._record_summary(SimpleNamespace(exposure_average_s=(index+1)/1000), 1/125)
            restarted = self.app(path)
            restarted._restore_summary()
            self.assertEqual(list(restarted.test_history[1/125]), [(i+1)/1000 for i in range(5, 25)])
            self.assertEqual(list(restarted.test_pass_history[1/125]), [bool(i % 2) for i in range(5, 25)])

    def test_explicit_clear_stays_cleared_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sheet.json"
            app = self.app(path)
            app._record_summary(SimpleNamespace(exposure_average_s=0.008), 1/125)
            app._clear_summary()
            restarted = self.app(path)
            restarted._restore_summary()
            self.assertEqual(restarted.test_history, {})
            self.assertEqual(restarted.test_pass_history, {})

    def test_failed_save_preserves_previous_file_and_current_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sheet.json"
            app = self.app(path)
            app._record_summary(SimpleNamespace(exposure_average_s=0.008), 1/125)
            original = path.read_bytes()
            with patch.object(Path, "replace", side_effect=OSError("disk full")):
                app._record_summary(SimpleNamespace(exposure_average_s=0.007), 1/125)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(app.test_history[1/125]), [0.008, 0.007])
            self.assertIn("could not be saved: disk full", app.summary_storage_status_var.get())
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_unreadable_sheet_is_preserved_until_explicit_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sheet.json"
            path.write_text("incomplete JSON", encoding="utf-8")
            app = self.app(path)
            app._restore_summary()
            self.assertTrue(app.summary_save_blocked)
            self.assertIn("could not be loaded", app.summary_storage_status_var.get())
            app._record_summary(SimpleNamespace(exposure_average_s=0.008), 1/125)
            self.assertEqual(path.read_text(encoding="utf-8"), "incomplete JSON")
            app._clear_summary()
            self.assertFalse(app.summary_save_blocked)
            self.assertEqual(tester.load_summary_history(path)[:2], ({}, {}))
            self.assertEqual(app.summary_storage_status_var.get(), "")

    def test_saved_sheet_is_visible_immediately_in_new_ui(self):
        try:
            root = tester.tk.Tk()
        except tester.tk.TclError as exc:
            self.skipTest(f"Tk display/runtime unavailable: {str(exc).splitlines()[0]}")
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as directory, patch.object(tester, "sd", None):
                path = Path(directory) / "sheet.json"
                app = self.app(path)
                app._record_summary(SimpleNamespace(exposure_average_s=0.008), 1/125)
                app._record_summary(SimpleNamespace(exposure_average_s=0.0079), 1/125)
                restarted = tester.ShutterTesterApp(root, summary_path=path)
                self.assertEqual(restarted.summary_count_var.get(), "2")
                row_id, = restarted.summary_table.get_children()
                self.assertEqual(restarted.summary_table.item(row_id, "values")[:2], ("1/125", "2/2"))
                self.assertEqual(restarted.speed_buttons["1/125"].cget("text"), "✓ 1/125")
                self.assertEqual(list(restarted.test_history[1/125]), [0.008, 0.0079])
        finally:
            root.destroy()


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


class LightIntegrationTests(unittest.TestCase):
    """Known optical durations through independently specified circular footprints."""

    def optical_recording(self, duration, coupling="AC coupled", gains=(0.35, 0.60)):
        fs = tester.FS
        times = np.arange(int((duration + 0.05) * fs)) / fs
        channels = []
        rng = np.random.default_rng(2026)
        for c, (crossing, delay, polarity, cutoff, tau) in enumerate((
            (0.00030, 0.014, -1, 2.5, 0.000015),
            (0.00016, 0.016625, 1, 4.0, 0.000035),
        )):
            def circle_fraction(position):
                z = np.clip(2 * position - 1, -1, 1)
                return (np.arcsin(z) + z * np.sqrt(np.maximum(0, 1-z*z)) + np.pi/2) / np.pi
            light = (circle_fraction((times-delay)/crossing + 0.5)
                     - circle_fraction((times-delay-duration)/crossing + 0.5))
            signal = tester.one_pole_lowpass(light, fs, tau)
            if coupling.startswith("AC"):
                signal = tester.one_pole_highpass(signal, fs, cutoff)
            channels.append(polarity*gains[c]*signal + rng.normal(0, 0.00008, len(times)))
        return np.column_stack(channels)

    def reference(self, coupling="AC coupled"):
        return tester.capture_integration_reference(
            self.optical_recording(1/30, coupling), 1/30, 3.3, 24, 2, coupling)

    def measure(self, samples, nominal, reference, coupling="AC coupled", **kwargs):
        return tester.analyze_signal(samples, nominal, 3.3, 24, 21, 2, coupling,
                                     ac_method=tester.INTEGRATION_METHOD,
                                     integration_reference=reference, **kwargs)

    def test_recovers_known_exposure_with_round_apertures_and_unequal_channels(self):
        for coupling in ("AC coupled", "DC / direct"):
            reference = self.reference(coupling)
            if coupling.startswith("AC"):
                for actual, expected in zip(reference.channels, (2.5, 4.0)):
                    self.assertAlmostEqual(actual.ac_cutoff_hz, expected, delta=0.15)
            for duration in (1/60, 1/1000, 1/2000, 1/4000, 1/8000):
                with self.subTest(coupling=coupling, duration=duration):
                    result = self.measure(self.optical_recording(duration, coupling), duration, reference, coupling)
                    self.assertEqual(result.method, tester.INTEGRATION_METHOD)
                    self.assertEqual(tester.infer_sensor_channels(result.event_indices), (1, 2))
                    for measured in (result.exposure_sensor_1_s, result.exposure_sensor_2_s):
                        self.assertAlmostEqual(measured, duration, delta=max(3e-6, duration*0.03))

    def test_exposure_is_area_and_markers_remain_curtain_edge_measurements(self):
        samples = self.optical_recording(1/8000)
        reference = self.reference()
        result = self.measure(samples, 1/4000, reference)
        edge_width = (result.event_indices["sensor1_close"]-result.event_indices["sensor1_open"])/tester.FS
        self.assertGreater(edge_width, 1/8000*1.4)
        self.assertAlmostEqual(result.exposure_sensor_1_s, 1/8000, delta=4e-6)
        self.assertEqual(result.integration_diagnostics["channels"][0]["edge_exposure_s"], edge_width)
        edges = tester.analyze_signal(samples, 1/4000, 3.3, 24, 21, 2)
        self.assertEqual(result.event_indices, edges.event_indices)
        self.assertEqual(result.first_curtain_full_travel_s, edges.first_curtain_full_travel_s)

    def test_requires_reference_and_matching_signal_mode_and_rate(self):
        samples = self.optical_recording(1/4000)
        with self.assertRaisesRegex(ValueError, "full-open reference"):
            self.measure(samples, 1/4000, None)
        reference = self.reference()
        with self.assertRaisesRegex(ValueError, "sample rate or signal mode"):
            self.measure(samples, 1/4000, replace(reference, sample_rate_hz=48000))
        with self.assertRaisesRegex(ValueError, "sample rate or signal mode"):
            self.measure(samples, 1/4000, reference, "DC / direct")

    def test_rejects_short_noisy_or_clipped_references(self):
        with self.assertRaisesRegex(ValueError, "1/60 or slower"):
            tester.capture_integration_reference(self.optical_recording(1/4000), 1/4000, 3.3, 24, 2)
        with self.assertRaises(ValueError):
            tester.capture_integration_reference(np.zeros((tester.FS//10, 2)), 1/60, 3.3, 24, 2)
        clipped = self.optical_recording(1/30)
        clipped[1500, 0] = -1
        with self.assertRaisesRegex(ValueError, "clips"):
            tester.capture_integration_reference(clipped, 1/30, 3.3, 24, 2)
        flicker = self.optical_recording(1/30)
        flicker[2200:3000] *= 1.5
        with self.assertRaises(ValueError):
            tester.capture_integration_reference(flicker, 1/30, 3.3, 24, 2)

    def test_detects_resolved_light_drift_and_inconsistent_reconstruction(self):
        reference = self.reference()
        with self.assertRaisesRegex(ValueError, "changed by over 10%"):
            self.measure(self.optical_recording(1/60)*0.6, 1/60, reference)
        with self.assertRaisesRegex(ValueError, "exceeds the full-open reference"):
            self.measure(self.optical_recording(1/60)*1.25, 1/60, reference)
        bad = replace(reference, channels=(replace(reference.channels[0], ac_cutoff_hz=300), reference.channels[1]))
        with self.assertRaises(ValueError):
            self.measure(self.optical_recording(1/4000), 1/4000, bad)

    def test_fast_area_cannot_distinguish_dimmer_light_from_less_exposure(self):
        # A fundamental reference condition, not a property that software can verify
        # from an unresolved pulse without an independent illumination monitor.
        duration = 1/8000
        result = self.measure(self.optical_recording(duration)*0.6, duration, self.reference())
        self.assertAlmostEqual(result.exposure_average_s, duration*0.6, delta=4e-6)

    def test_reference_and_area_debug_report_round_trip(self):
        duration = 1/4000
        reference = self.reference()
        samples = self.optical_recording(duration)
        result = self.measure(samples, duration, reference)
        settings = dict(sample_rate_hz=tester.FS, nominal_s=duration, reference_distance_mm=24,
                        integration_reference=asdict(reference), capture_purpose="measurement")
        with tempfile.TemporaryDirectory() as directory:
            file = tester.write_debug_capture(directory, samples, samples, settings, result)
            report = json.loads(file.with_suffix(".json").read_text(encoding="utf-8"))
            replay = self.measure(samples, duration, report["settings"]["integration_reference"])
            self.assertEqual(replay.integration_diagnostics, report["result"]["integration_diagnostics"])
            self.assertEqual(replay.exposure_average_s, result.exposure_average_s)
            settings["capture_purpose"] = "full-open reference"
            file = tester.write_debug_capture(directory, samples, samples, settings)
            self.assertEqual(json.loads(file.with_suffix(".json").read_text(encoding="utf-8"))["outcome"], "reference")

    def test_reference_ui_switches_method_and_preserves_summary_across_references(self):
        try:
            root = tester.tk.Tk()
        except tester.tk.TclError as exc:
            self.skipTest(f"Tk display/runtime unavailable: {str(exc).splitlines()[0]}")
        root.withdraw()
        simulate = tester.simulate_shutter
        try:
            with tempfile.TemporaryDirectory() as directory, patch.object(tester, "sd", None), patch.object(tester.messagebox, "showerror") as popup:
                with patch.object(tester, "simulate_shutter", side_effect=lambda *a, **k:
                                  simulate(*a, **k, rng=np.random.default_rng(5))):
                    summary_path = Path(directory) / "summary.state"
                    app = tester.ShutterTesterApp(root, summary_path=summary_path)
                    app.debug_directory = Path(directory)
                    app.nominal_var.set("1/4000")
                    app.run_simulator()
                    self.assertEqual(len(app.test_history[1/4000]), 1)
                    app.debug_mode_var.set(True)
                    app._request_reference_capture()
                    popup.assert_not_called()
                    self.assertIsNotNone(app.integration_reference)
                    self.assertFalse(app.reference_pending)
                    self.assertEqual(app.nominal_var.get(), "1/4000")
                    self.assertEqual(app.ac_method_var.get(), tester.INTEGRATION_METHOD)
                    self.assertEqual(len(app.test_history[1/4000]), 1)
                    reports = [json.loads(f.read_text(encoding="utf-8")) for f in Path(directory).glob("*.json")]
                    self.assertEqual(len(reports), 1)
                    self.assertEqual(reports[0]["outcome"], "reference")
                    self.assertEqual(reports[0]["settings"]["nominal_speed"], "1/60")
                    app.run_simulator()
                    popup.assert_not_called()
                    self.assertEqual(len(app.test_history[1/4000]), 2)
                    self.assertEqual(len(app.summary_table.get_children()), 1)
                    original_history = list(app.test_history[1/4000])
                    original_verdicts = list(app.test_pass_history[1/4000])
                    first_reference_id = app.integration_reference.reference_id
                    self.assertIn("light area", app.ax.get_title())
                    bad = simulate(1/4000, 3.3, 3.3, 24, 21, 2, rng=np.random.default_rng(5)) * 4
                    with self.assertRaisesRegex(ValueError, "clips"):
                        app._analyze_and_display(bad, "simulator")
                    self.assertIsNone(app.result)
                    self.assertEqual(app.pass_fail_var.get(), "NOT TESTED")
                    self.assertEqual(app.exposure_status_vars["average"].get(), "REJECTED")
                    self.assertEqual(len(app.test_history[1/4000]), 2)
                    with patch.object(app, "run_simulator"):
                        app._request_reference_capture()
                        self.assertTrue(app.reference_pending)
                        self.assertEqual(len(app.summary_table.get_children()), 1)
                        app._request_reference_capture()  # Cancel while waiting for a shot.
                    self.assertFalse(app.reference_pending)
                    self.assertEqual(list(app.test_history[1/4000]), original_history)
                    app._request_reference_capture()
                    popup.assert_not_called()
                    self.assertNotEqual(app.integration_reference.reference_id, first_reference_id)
                    self.assertEqual(list(app.test_history[1/4000]), original_history)
                    self.assertEqual(list(app.test_pass_history[1/4000]), original_verdicts)
                    app.run_simulator()
                    popup.assert_not_called()
                    self.assertEqual(len(app.test_history[1/4000]), 3)
                    self.assertEqual(len(app.test_pass_history[1/4000]), 3)
                    row_id, = app.summary_table.get_children()
                    self.assertEqual(app.summary_table.item(row_id, "values")[1], "3/5")
                    app._clear_integration_reference()
                    self.assertEqual(len(app.test_history[1/4000]), 3)
                    self.assertEqual(len(app.summary_table.get_children()), 1)
                    app._request_reference_capture()
                    app.aperture_var.set("1.5")
                    self.assertIsNone(app.integration_reference)
                    self.assertIn("new reference", app.reference_status_var.get())
                    self.assertEqual(len(app.test_history[1/4000]), 3)
                    saved_history = list(app.test_history[1/4000])
                    for method in (tester.DEFAULT_AC_METHOD, "AC pulse fit", tester.INTEGRATION_METHOD):
                        app.ac_method_var.set(method)
                        app._analysis_method_changed()
                        self.assertEqual(list(app.test_history[1/4000]), saved_history)
                        self.assertEqual(len(app.summary_table.get_children()), 1)
                    for var, value in ((app.camera_var, "Nikon F80 / N80"),
                                       (app.spacing_var, "20"), (app.path_var, "23"),
                                       (app.expected_travel_var, "4"),
                                       (app.coupling_var, "DC / direct coupled"),
                                       (app.source_var, "Live test input")):
                        var.set(value)
                        self.assertEqual(list(app.test_history[1/4000]), saved_history)
                        self.assertEqual(len(app.summary_table.get_children()), 1)
                    app._source_changed()
                    self.assertEqual(list(app.test_history[1/4000]), saved_history)
                    app.summary_count_var.set("1")
                    row_id, = app.summary_table.get_children()
                    self.assertEqual(app.summary_table.item(row_id, "values")[1], "1/1")
                    self.assertEqual(list(app.test_history[1/4000]), saved_history)
                    restored_history, restored_verdicts, restored_count = tester.load_summary_history(summary_path)
                    self.assertEqual(list(restored_history[1/4000]), saved_history)
                    self.assertEqual(list(restored_verdicts[1/4000]), list(app.test_pass_history[1/4000]))
                    self.assertEqual(restored_count, 1)
                    app._clear_summary()
                    self.assertEqual(app.test_history, {})
                    self.assertEqual(app.test_pass_history, {})
                    self.assertEqual(app.summary_table.get_children(), ())
                    self.assertEqual(tester.load_summary_history(summary_path)[:2], ({}, {}))
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
