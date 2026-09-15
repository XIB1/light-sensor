#!/usr/bin/env python3
"""
Two-point camera shutter tester.

Input sources
-------------
* Built-in shutter simulator
* Any PortAudio recording device exposed by the `sounddevice` package

Live input
----------
A selected audio device is opened as a CONTINUOUS stereo stream. Audio is
always written into a rolling ring buffer. Pressing "ARM LIVE INPUT" tells the
program to detect the next shutter event in that already-running stream, so the
first-curtain transient is not lost while the sound device is being opened.

For shutter speeds where opening/closing transients should be separable, live
capture uses a small state machine:

    ARMED -> OPEN DETECTED -> WAITING FOR CLOSE -> ANALYZE

The close deadline is based on the selected shutter speed. It allows extra
slow-side error for initial tensioning: +2 stops below 1/60, +3 stops from
1/60 through 1/1000, and +4 stops from 1/2000 upward, plus curtain-travel
and processing margin. If a valid closing event is not detected on both channels before the
deadline, the measurement ends as INCOMPLETE / TIMEOUT. It is NOT reported as
a shutter PASS or FAIL.

At the fastest speeds, where the sensor waveform is a short overlapping pulse
rather than distinct opening/closing transients, the tester captures a fixed
post-trigger window and uses the existing template-fit analyzer instead.

Dependencies
------------
    pip install numpy matplotlib sounddevice

The simulator still runs if `sounddevice` is not installed.

Run
---
    python shutter_tester_v6.py

Optional internal test:
    python shutter_tester_v6.py --self-test
"""

from __future__ import annotations

import math
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError as exc:
    raise SystemExit(
        "Missing matplotlib. Install with: pip install numpy matplotlib sounddevice"
    ) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PREFERRED_FS = 96_000

# 35 mm vertical-travel shutter dimension. Change to 36 mm for a horizontal
# shutter if the sensors are positioned along that travel direction.
GATE_MM = 24.0

CURTAIN_TRAVEL_MS = 3.3

# Effective optical footprint of each assembled sensor.
SENSOR_WIDTH_MM = 2.0

# Simulator analog response.
AC_COUPLING_HZ = 35.0
SENSOR_TAU_US = 7.0
NOISE_RMS = 0.003

# Live capture.
LIVE_BUFFER_SECONDS = 6.0
LIVE_PRETRIGGER_SECONDS = 0.012
LIVE_BASELINE_SECONDS = 0.150
LIVE_TRIGGER_SIGMA = 8.0
LIVE_MIN_TRIGGER = 0.008
LIVE_POLL_MS = 20
LIVE_POSTROLL_SECONDS = 0.012
LIVE_FAST_CAPTURE_SECONDS = 0.040

COMMON_SPEEDS = [
    "1", "1/2", "1/4", "1/8", "1/15", "1/30", "1/60", "1/125",
    "1/250", "1/500", "1/1000", "1/2000", "1/4000", "1/8000",
]

TOLERANCE_OPTIONS = {
    "±1/6 stop": 1.0 / 6.0,
    "±1/3 stop": 1.0 / 3.0,
    "±1/2 stop": 1.0 / 2.0,
    "±1 stop": 1.0,
}

TRAVEL_TOLERANCE_OPTIONS = {
    "±2%": 2.0,
    "±5%": 5.0,
    "±10%": 10.0,
    "±15%": 15.0,
    "±20%": 20.0,
}

TRAVEL_TIME_PRESETS_MS = (2.5, 3.0, 3.3, 3.5, 4.0, 5.0, 8.0, 10.0, 12.0)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def parse_speed(text: str) -> float:
    text = text.strip()
    if "/" in text:
        a, b = text.split("/", 1)
        value = float(a) / float(b)
    else:
        value = float(text)
    if value <= 0:
        raise ValueError("Exposure must be positive.")
    return value


def format_speed(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds <= 0:
        return "—"
    if seconds >= 0.8:
        return f"{seconds:.3f} s"
    denom = 1.0 / seconds
    if denom < 10:
        return f"1/{denom:.2f} s"
    return f"1/{denom:.0f} s"


def one_pole_lowpass(x: np.ndarray, fs: float, tau_s: float) -> np.ndarray:
    if tau_s <= 0:
        return x.copy()
    alpha = 1.0 - math.exp(-1.0 / (fs * tau_s))
    y = np.empty_like(x, dtype=np.float64)
    y[0] = alpha * x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
    return y


def one_pole_highpass(x: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return x.copy()

    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / fs
    alpha = rc / (rc + dt)

    y = np.zeros_like(x, dtype=np.float64)
    for i in range(1, len(x)):
        y[i] = alpha * (y[i - 1] + x[i] - x[i - 1])
    return y


def stops_from_nominal(measured_s: float, nominal_s: float) -> float:
    """
    Positive = longer than nominal = shutter slow / more exposure.
    Negative = shorter than nominal = shutter fast / less exposure.
    """
    if measured_s <= 0 or nominal_s <= 0:
        return float("nan")
    return math.log2(measured_s / nominal_s)


def format_stop_error(stops: float) -> str:
    if not np.isfinite(stops):
        return "—"
    if abs(stops) < 0.005:
        return "0.00 stops"
    direction = "slow / more exposure" if stops > 0 else "fast / less exposure"
    return f"{stops:+.2f} stops ({direction})"


def curtain_speed_m_s(travel_s: float, gate_mm: float = GATE_MM) -> float:
    if travel_s <= 0:
        return float("nan")
    return (gate_mm / 1000.0) / travel_s


def travel_error_percent(measured_s: float, expected_s: float) -> float:
    """Positive = curtain faster than expected, negative = slower."""
    if measured_s <= 0 or expected_s <= 0:
        return float("nan")
    measured_speed = curtain_speed_m_s(measured_s)
    expected_speed = curtain_speed_m_s(expected_s)
    return 100.0 * (measured_speed / expected_speed - 1.0)


def format_travel_measurement(measured_s: float, expected_s: float) -> str:
    if not np.isfinite(measured_s) or measured_s <= 0:
        return "—"
    speed = curtain_speed_m_s(measured_s)
    error = travel_error_percent(measured_s, expected_s)
    direction = "fast" if error > 0 else "slow" if error < 0 else "on target"
    return (
        f"{measured_s * 1000:.3f} ms  |  {speed:.2f} m/s  |  "
        f"speed {error:+.1f}% ({direction})"
    )


def sensor_crossing_time_s(expected_travel_ms: float) -> float:
    velocity_mm_s = GATE_MM / (expected_travel_ms / 1000.0)
    return SENSOR_WIDTH_MM / velocity_mm_s


def uses_transient_method(nominal_s: float, expected_travel_ms: float) -> bool:
    """Keep the live-input regime identical to analyze_signal()."""
    crossing = sensor_crossing_time_s(expected_travel_ms)
    return nominal_s / crossing >= 1.8


def live_timeout_slow_stops(nominal_s: float) -> float:
    """
    Slow-side timeout allowance for initial shutter adjustment.

    Slower than 1/60:      +2 stops  (4x nominal)
    1/60 through 1/1000:  +3 stops  (8x nominal)
    1/2000 and faster:     +4 stops (16x nominal)

    The selected thresholds intentionally leave extra room at high speeds,
    where an initially mistensioned shutter can be many stops slow.
    """
    if nominal_s <= 1.0 / 2000.0:
        return 4.0
    if nominal_s <= 1.0 / 60.0:
        return 3.0
    return 2.0


def live_close_timeout_s(nominal_s: float, expected_travel_ms: float) -> float:
    """
    Maximum time after first opening detection to wait for closing transients.

    Timeout is based on the selected shutter speed, with extra slow-side
    allowance for initial tensioning. Add two expected curtain-travel times
    and 25 ms of processing/mechanical margin. A 50 ms floor keeps very fast
    settings practical and gives the detector enough samples.
    """
    expected_travel_s = expected_travel_ms / 1000.0
    slow_stops = live_timeout_slow_stops(nominal_s)
    slow_multiplier = 2.0 ** slow_stops
    return max(
        0.050,
        slow_multiplier * nominal_s + 2.0 * expected_travel_s + 0.025,
    )


# ---------------------------------------------------------------------------
# Physical-ish simulator
# ---------------------------------------------------------------------------

def finite_sensor_light_signal(
    t: np.ndarray,
    x_start_mm: float,
    x_end_mm: float,
    first_curtain_speed_mm_s: float,
    second_curtain_speed_mm_s: float,
    exposure_s: float,
    spatial_samples: int = 160,
) -> np.ndarray:
    width = x_end_mm - x_start_mm
    dx = width / spatial_samples
    xs = x_start_mm + (np.arange(spatial_samples) + 0.5) * dx

    t_open = xs / first_curtain_speed_mm_s
    t_close = exposure_s + xs / second_curtain_speed_mm_s

    lit = (
        (t[:, None] >= t_open[None, :])
        & (t[:, None] < t_close[None, :])
    )
    return lit.mean(axis=1, dtype=np.float64)


def simulate_shutter(
    exposure_s: float,
    fs: int = PREFERRED_FS,
    gate_mm: float = GATE_MM,
    travel_ms: float = CURTAIN_TRAVEL_MS,
    second_travel_ms: float | None = None,
    sensor_width_mm: float = SENSOR_WIDTH_MM,
    ac_hz: float = AC_COUPLING_HZ,
    sensor_tau_us: float = SENSOR_TAU_US,
    noise_rms: float = NOISE_RMS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    if rng is None:
        rng = np.random.default_rng()

    if second_travel_ms is None:
        second_travel_ms = travel_ms

    first_travel_s = travel_ms / 1000.0
    second_travel_s = second_travel_ms / 1000.0
    if first_travel_s <= 0 or second_travel_s <= 0:
        raise ValueError("Curtain travel times must be positive.")

    first_velocity = gate_mm / first_travel_s
    second_velocity = gate_mm / second_travel_s

    # Footprints just inside the two opposite edges.
    s1 = (0.0, sensor_width_mm)
    s2 = (gate_mm - sensor_width_mm, gate_mm)

    pre_s = 0.012
    post_s = max(0.030, 5.0 / (2.0 * math.pi * ac_hz)) if ac_hz > 0 else 0.030
    total_s = pre_s + max(first_travel_s, second_travel_s) + exposure_s + post_s

    n = int(math.ceil(total_s * fs))
    t = np.arange(n, dtype=np.float64) / fs - pre_s

    channels = []
    for x0, x1 in (s1, s2):
        light = finite_sensor_light_signal(
            t,
            x0,
            x1,
            first_velocity,
            second_velocity,
            exposure_s,
        )
        analog = one_pole_lowpass(light, fs, sensor_tau_us * 1e-6)
        recorded = one_pole_highpass(analog, fs, ac_hz)
        recorded *= 0.82

        if noise_rms > 0:
            recorded += rng.normal(0.0, noise_rms, len(recorded))

        channels.append(recorded)

    stereo = np.column_stack(channels).astype(np.float32)

    peak = np.max(np.abs(stereo))
    if peak > 0.98:
        stereo *= 0.98 / peak

    center_spacing_mm = gate_mm - sensor_width_mm
    meta = {
        "exposure_s": exposure_s,
        "first_velocity_mm_s": first_velocity,
        "second_velocity_mm_s": second_velocity,
        "center_spacing_mm": center_spacing_mm,
        "first_travel_s": first_travel_s,
        "second_travel_s": second_travel_s,
    }
    return stereo, meta


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    method: str
    exposure_sensor_1_s: float
    exposure_sensor_2_s: float
    exposure_average_s: float
    first_curtain_sensor_delay_s: float
    second_curtain_sensor_delay_s: float
    first_curtain_full_travel_s: float
    second_curtain_full_travel_s: float
    sensor_mismatch_percent: float
    confidence: float
    event_indices: dict


def robust_noise_sigma(x: np.ndarray, fs: int) -> float:
    n = min(len(x), max(32, int(fs * 0.008)))
    baseline = x[:n]
    med = np.median(baseline)
    mad = np.median(np.abs(baseline - med))
    return max(1e-7, 1.4826 * mad)


def smooth(x: np.ndarray, width: int = 5) -> np.ndarray:
    if width <= 1:
        return x.astype(np.float64, copy=True)
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(x, kernel, mode="same")


def transient_measurement(
    channel: np.ndarray,
    fs: int,
    nominal_s: float,
) -> tuple[float, int, int, float]:
    x = smooth(channel.astype(np.float64), 5)
    sigma = robust_noise_sigma(x, fs)

    start = int(0.004 * fs)
    open_end = min(len(x), start + int(0.020 * fs))
    if open_end <= start + 4:
        raise ValueError("Signal buffer too short.")

    open_idx = start + int(np.argmax(x[start:open_end]))
    open_amp = x[open_idx]

    low_s = max(0.00005, nominal_s / 4.0)
    high_s = nominal_s * 4.0

    close_start = max(open_idx + 2, open_idx + int(low_s * fs))
    close_end = min(len(x), open_idx + int(high_s * fs) + int(0.012 * fs))

    if close_end <= close_start + 4:
        raise ValueError("Could not form closing-edge search window.")

    close_idx = close_start + int(np.argmin(x[close_start:close_end]))
    close_amp = -x[close_idx]

    measured_s = (close_idx - open_idx) / fs

    snr_like = min(open_amp, close_amp) / sigma
    confidence = float(np.clip((snr_like - 3.0) / 20.0, 0.0, 1.0))

    return measured_s, open_idx, close_idx, confidence


def make_single_sensor_template(
    exposure_s: float,
    fs: int,
    travel_ms: float,
    sensor_width_mm: float,
    ac_hz: float,
    sensor_tau_us: float,
) -> np.ndarray:
    velocity = GATE_MM / (travel_ms / 1000.0)
    pre_s = 0.0015
    post_s = (
        max(0.003, min(0.012, 2.0 / (2.0 * math.pi * ac_hz)))
        if ac_hz > 0 else 0.004
    )
    total_s = pre_s + sensor_width_mm / velocity + exposure_s + post_s

    n = max(32, int(math.ceil(total_s * fs)))
    t = np.arange(n) / fs - pre_s

    light = finite_sensor_light_signal(
        t,
        0.0,
        sensor_width_mm,
        velocity,
        velocity,
        exposure_s,
        spatial_samples=120,
    )
    y = one_pole_lowpass(light, fs, sensor_tau_us * 1e-6)
    y = one_pole_highpass(y, fs, ac_hz)

    y -= np.mean(y)
    norm = np.linalg.norm(y)
    if norm > 0:
        y /= norm
    return y


def best_normalized_match(signal: np.ndarray, template: np.ndarray) -> tuple[float, int]:
    x = signal.astype(np.float64)
    t = template.astype(np.float64)

    if len(t) >= len(x):
        return -1.0, 0

    t = t - np.mean(t)
    tnorm = np.linalg.norm(t)
    if tnorm == 0:
        return -1.0, 0
    t /= tnorm

    corr = np.correlate(x, t, mode="valid")

    n = len(t)
    cs = np.concatenate(([0.0], np.cumsum(x)))
    cs2 = np.concatenate(([0.0], np.cumsum(x * x)))
    sums = cs[n:] - cs[:-n]
    sums2 = cs2[n:] - cs2[:-n]
    var_energy = sums2 - (sums * sums) / n
    denom = np.sqrt(np.maximum(var_energy, 1e-12))

    score = corr / denom
    idx = int(np.argmax(score))
    return float(score[idx]), idx


def template_measurement(
    channel: np.ndarray,
    fs: int,
    nominal_s: float,
    expected_travel_ms: float,
) -> tuple[float, int, float]:
    start = int(0.003 * fs)
    end = min(len(channel), start + int(0.030 * fs))
    segment = channel[start:end].astype(np.float64)

    # +/-2 stops.
    candidates = nominal_s * (2.0 ** np.linspace(-2.0, 2.0, 121))

    best_score = -1e9
    best_s = nominal_s
    best_idx = 0

    for exposure_s in candidates:
        template = make_single_sensor_template(
            float(exposure_s),
            fs,
            expected_travel_ms,
            SENSOR_WIDTH_MM,
            AC_COUPLING_HZ,
            SENSOR_TAU_US,
        )
        score, idx = best_normalized_match(segment, template)

        if score > best_score:
            best_score = score
            best_s = float(exposure_s)
            best_idx = start + idx

    confidence = float(np.clip(best_score, 0.0, 1.0))
    return best_s, best_idx, confidence


def analyze_signal(
    stereo: np.ndarray,
    nominal_s: float,
    fs: int = PREFERRED_FS,
    expected_travel_ms: float = CURTAIN_TRAVEL_MS,
) -> AnalysisResult:
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        raise ValueError("Expected stereo samples with shape (N, 2).")
    if expected_travel_ms <= 0:
        raise ValueError("Expected curtain travel time must be positive.")

    use_transients = uses_transient_method(nominal_s, expected_travel_ms)

    measurements = []
    event_indices = {}
    confidences = []

    if use_transients:
        for c in range(2):
            exp_s, open_i, close_i, conf = transient_measurement(
                stereo[:, c], fs, nominal_s
            )
            measurements.append(exp_s)
            event_indices[f"sensor{c+1}_open"] = open_i
            event_indices[f"sensor{c+1}_close"] = close_i
            confidences.append(conf)

        method = "transient edges"

    else:
        template_starts = []
        for c in range(2):
            exp_s, start_i, conf = template_measurement(
                stereo[:, c], fs, nominal_s, expected_travel_ms
            )
            measurements.append(exp_s)
            template_starts.append(start_i)
            confidences.append(conf)

        event_indices["sensor1_open"] = template_starts[0]
        event_indices["sensor2_open"] = template_starts[1]
        event_indices["sensor1_close"] = (
            template_starts[0] + int(measurements[0] * fs)
        )
        event_indices["sensor2_close"] = (
            template_starts[1] + int(measurements[1] * fs)
        )

        method = "template fit"

    exp1, exp2 = measurements
    avg = (exp1 + exp2) / 2.0

    o1 = event_indices["sensor1_open"]
    o2 = event_indices["sensor2_open"]
    c1 = event_indices["sensor1_close"]
    c2 = event_indices["sensor2_close"]

    first_delay = (o2 - o1) / fs
    second_delay = (c2 - c1) / fs

    center_spacing_mm = GATE_MM - SENSOR_WIDTH_MM

    def delay_to_full_travel(delay_s: float) -> float:
        if abs(delay_s) < 1.0 / fs:
            return float("nan")
        velocity_measured = center_spacing_mm / abs(delay_s)
        return GATE_MM / velocity_measured

    first_full = delay_to_full_travel(first_delay)
    second_full = delay_to_full_travel(second_delay)

    mismatch = 100.0 * abs(exp1 - exp2) / avg if avg > 0 else float("nan")

    return AnalysisResult(
        method=method,
        exposure_sensor_1_s=exp1,
        exposure_sensor_2_s=exp2,
        exposure_average_s=avg,
        first_curtain_sensor_delay_s=first_delay,
        second_curtain_sensor_delay_s=second_delay,
        first_curtain_full_travel_s=first_full,
        second_curtain_full_travel_s=second_full,
        sensor_mismatch_percent=mismatch,
        confidence=float(np.mean(confidences)),
        event_indices=event_indices,
    )


# ---------------------------------------------------------------------------
# Continuous live-audio ring buffer
# ---------------------------------------------------------------------------

class StereoRingBuffer:
    """Thread-safe stereo ring buffer using absolute sample positions."""

    def __init__(self, capacity_frames: int):
        self.capacity = max(16, int(capacity_frames))
        self.data = np.zeros((self.capacity, 2), dtype=np.float32)
        self.total_written = 0
        self.lock = threading.Lock()

    def write(self, frames: np.ndarray) -> None:
        x = np.asarray(frames, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] < 2 or len(x) == 0:
            return
        x = x[:, :2]

        if len(x) >= self.capacity:
            x = x[-self.capacity:]

        with self.lock:
            start_abs = self.total_written
            start = start_abs % self.capacity
            n = len(x)
            first = min(n, self.capacity - start)

            self.data[start:start + first] = x[:first]
            if first < n:
                self.data[:n - first] = x[first:]

            self.total_written += n

    def bounds(self) -> tuple[int, int]:
        with self.lock:
            newest = self.total_written
            oldest = max(0, newest - self.capacity)
            return oldest, newest

    def get(self, start_abs: int, end_abs: int) -> np.ndarray:
        if end_abs < start_abs:
            raise ValueError("Invalid ring-buffer range.")

        with self.lock:
            newest = self.total_written
            oldest = max(0, newest - self.capacity)

            if start_abs < oldest or end_abs > newest:
                raise ValueError(
                    f"Requested live samples {start_abs}:{end_abs} are outside "
                    f"the retained buffer {oldest}:{newest}."
                )

            n = end_abs - start_abs
            if n == 0:
                return np.empty((0, 2), dtype=np.float32)

            start = start_abs % self.capacity
            first = min(n, self.capacity - start)

            if first == n:
                return self.data[start:start + n].copy()

            return np.vstack(
                (
                    self.data[start:].copy(),
                    self.data[:n - first].copy(),
                )
            )

    def last(self, frames: int) -> np.ndarray:
        oldest, newest = self.bounds()
        start = max(oldest, newest - int(frames))
        return self.get(start, newest)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class ShutterTesterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Two-Point Shutter Tester")

        # Fit the initial window to the available display.  The controls
        # sidebar itself is scrollable, so the application remains usable on
        # smaller displays and with Windows display scaling.
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        initial_w = max(900, min(1360, screen_w - 80))
        initial_h = max(620, min(800, screen_h - 100))
        root.geometry(f"{initial_w}x{initial_h}")
        root.minsize(min(1000, initial_w), min(620, initial_h))

        self.samples: np.ndarray | None = None
        self.sample_rate = PREFERRED_FS
        self.meta: dict | None = None
        self.result: AnalysisResult | None = None
        self.plot_source_label = "simulator"

        # Simulator/settings state.
        self.nominal_var = tk.StringVar(value="1/1000")
        self.actual_var = tk.StringVar(value="1/1000")
        self.follow_nominal_var = tk.BooleanVar(value=True)
        self.tolerance_var = tk.StringVar(value="±1/3 stop")
        self.expected_travel_var = tk.StringVar(value=f"{CURTAIN_TRAVEL_MS:g}")
        self.travel_tolerance_var = tk.StringVar(value="±10%")
        self.sim_first_travel_var = tk.StringVar(value=f"{CURTAIN_TRAVEL_MS:g}")
        self.sim_second_travel_var = tk.StringVar(value=f"{CURTAIN_TRAVEL_MS:g}")
        self.follow_expected_travel_var = tk.BooleanVar(value=True)

        # Input state.
        self.input_var = tk.StringVar(value="Simulator")
        self.input_status_var = tk.StringVar(value="Simulator selected")
        self.device_map: dict[str, int | None] = {"Simulator": None}

        self.audio_stream = None
        self.audio_buffer: StereoRingBuffer | None = None
        self.audio_fs: int | None = None
        self.audio_device_index: int | None = None
        self.audio_stream_error: str | None = None

        # Live measurement state.
        self.live_state = "idle"
        self.live_scan_index = 0
        self.live_first_open_index: int | None = None
        self.live_deadline_index: int | None = None
        self.live_capture_end_index: int | None = None
        self.live_open_seen = [False, False]
        self.live_close_seen = [False, False]
        self.live_open_indices: list[int | None] = [None, None]
        self.live_close_indices: list[int | None] = [None, None]
        self.live_baseline = np.zeros(2, dtype=np.float64)
        self.live_threshold = np.full(2, LIVE_MIN_TRIGGER, dtype=np.float64)
        self.live_nominal_s = 0.001
        self.live_expected_travel_ms = CURTAIN_TRAVEL_MS
        self.live_transient_mode = True

        # Output vars.
        self.status_var = tk.StringVar(value="Ready — simulator input")
        self.method_var = tk.StringVar(value="—")
        self.avg_var = tk.StringVar(value="—")
        self.s1_var = tk.StringVar(value="—")
        self.s2_var = tk.StringVar(value="—")
        self.variance_var = tk.StringVar(value="—")
        self.s1_variance_var = tk.StringVar(value="—")
        self.s2_variance_var = tk.StringVar(value="—")
        self.travel1_var = tk.StringVar(value="—")
        self.travel2_var = tk.StringVar(value="—")
        self.mismatch_var = tk.StringVar(value="—")
        self.conf_var = tk.StringVar(value="—")
        self.pass_fail_var = tk.StringVar(value="NOT TESTED")
        self.tolerance_detail_var = tk.StringVar(
            value="Select a speed and run a test."
        )
        self.expected_speed_var = tk.StringVar(value="—")
        self.travel1_status_var = tk.StringVar(value="NOT TESTED")
        self.travel2_status_var = tk.StringVar(value="NOT TESTED")
        self.travel_detail_var = tk.StringVar(
            value="Set the expected full-gate travel time."
        )

        self._build_ui()
        self.refresh_input_devices()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(LIVE_POLL_MS, self._poll_live_input)
        self.root.after(100, self.run_test)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        # A little more horizontal room for controls, while keeping the
        # entire sidebar independent of the available vertical height.
        outer.columnconfigure(0, minsize=420)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        sidebar_shell = ttk.Frame(outer, width=420)
        sidebar_shell.grid(row=0, column=0, sticky="nsew")
        sidebar_shell.grid_propagate(False)
        sidebar_shell.rowconfigure(0, weight=1)
        sidebar_shell.columnconfigure(0, weight=1)

        main = ttk.Frame(outer, padding=(10, 0, 0, 0))
        main.grid(row=0, column=1, sticky="nsew")

        ttk.Separator(outer, orient="vertical").grid(
            row=0, column=0, sticky="nse"
        )

        # ---------------------------------------------------------------
        # Scrollable settings area.
        #
        # The action buttons are deliberately NOT inside this canvas;
        # they live in sidebar_footer below and therefore remain visible
        # even when the settings need to scroll.
        # ---------------------------------------------------------------
        scroll_area = ttk.Frame(sidebar_shell)
        scroll_area.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        scroll_area.rowconfigure(0, weight=1)
        scroll_area.columnconfigure(0, weight=1)

        sidebar_canvas = tk.Canvas(
            scroll_area,
            highlightthickness=0,
            borderwidth=0,
        )
        sidebar_scrollbar = ttk.Scrollbar(
            scroll_area,
            orient="vertical",
            command=sidebar_canvas.yview,
        )
        sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)

        sidebar_canvas.grid(row=0, column=0, sticky="nsew")
        sidebar_scrollbar.grid(row=0, column=1, sticky="ns")

        sidebar = ttk.Frame(sidebar_canvas, padding=(2, 0, 3, 2))
        sidebar_window = sidebar_canvas.create_window(
            (0, 0),
            window=sidebar,
            anchor="nw",
        )

        sidebar_overflows = {"value": False}

        def _update_sidebar_scroll_state():
            """
            Keep the settings pane pinned to the top unless its contents are
            genuinely taller than the visible canvas.  Only then allow
            scrolling.
            """
            bbox = sidebar_canvas.bbox("all")
            if bbox is None:
                return

            content_h = max(0, bbox[3] - bbox[1])
            viewport_h = max(1, sidebar_canvas.winfo_height())
            overflows = content_h > viewport_h + 2
            sidebar_overflows["value"] = overflows

            if overflows:
                sidebar_canvas.configure(scrollregion=bbox)
                sidebar_scrollbar.state(["!disabled"])
            else:
                # Pin the contents to the top and make the scroll region at
                # least as tall as the viewport.  This prevents Tk's canvas
                # from drifting vertically when the wheel is used.
                sidebar_canvas.yview_moveto(0.0)
                sidebar_canvas.configure(
                    scrollregion=(0, 0, max(1, sidebar_canvas.winfo_width()), viewport_h)
                )
                sidebar_scrollbar.state(["disabled"])

        def _sidebar_contents_changed(_event=None):
            _update_sidebar_scroll_state()

        def _sidebar_canvas_resized(event):
            # Make the embedded frame follow the canvas width exactly so
            # comboboxes and group boxes expand horizontally.
            sidebar_canvas.itemconfigure(
                sidebar_window,
                width=event.width,
            )
            # Re-evaluate whether scrolling is necessary whenever the window
            # is resized.
            self.root.after_idle(_update_sidebar_scroll_state)

        def _sidebar_mousewheel(event):
            # Ignore wheel input entirely when everything already fits.
            if not sidebar_overflows["value"]:
                sidebar_canvas.yview_moveto(0.0)
                return "break"

            # Windows/macOS mouse wheel.
            if getattr(event, "delta", 0):
                steps = -1 if event.delta > 0 else 1
                # Larger Windows deltas can represent multiple steps.
                if abs(event.delta) >= 120:
                    steps = -int(event.delta / 120)
                sidebar_canvas.yview_scroll(steps, "units")
            return "break"

        def _sidebar_linux_up(_event):
            if sidebar_overflows["value"]:
                sidebar_canvas.yview_scroll(-1, "units")
            else:
                sidebar_canvas.yview_moveto(0.0)
            return "break"

        def _sidebar_linux_down(_event):
            if sidebar_overflows["value"]:
                sidebar_canvas.yview_scroll(1, "units")
            else:
                sidebar_canvas.yview_moveto(0.0)
            return "break"

        sidebar.bind("<Configure>", _sidebar_contents_changed)
        sidebar_canvas.bind("<Configure>", _sidebar_canvas_resized)

        # Bind the wheel only while the pointer is over the settings area.
        def _enable_sidebar_wheel(_event):
            self.root.bind_all("<MouseWheel>", _sidebar_mousewheel)
            self.root.bind_all("<Button-4>", _sidebar_linux_up)
            self.root.bind_all("<Button-5>", _sidebar_linux_down)

        def _disable_sidebar_wheel(_event):
            self.root.unbind_all("<MouseWheel>")
            self.root.unbind_all("<Button-4>")
            self.root.unbind_all("<Button-5>")

        scroll_area.bind("<Enter>", _enable_sidebar_wheel)
        scroll_area.bind("<Leave>", _disable_sidebar_wheel)

        # Fixed footer: the main action buttons never disappear below
        # the bottom edge of the window.
        sidebar_footer = ttk.Frame(sidebar_shell, padding=(2, 6, 8, 0))
        sidebar_footer.grid(row=1, column=0, sticky="ew")
        sidebar_footer.columnconfigure(0, weight=1)

        # Input selector -------------------------------------------------
        input_frame = ttk.LabelFrame(sidebar, text="Input source", padding=5)
        input_frame.pack(fill="x")

        self.input_combo = ttk.Combobox(
            input_frame,
            textvariable=self.input_var,
            state="readonly",
        )
        self.input_combo.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.input_combo.bind("<<ComboboxSelected>>", self._input_changed)

        ttk.Button(
            input_frame,
            text="Refresh",
            command=self.refresh_input_devices,
            width=8,
        ).grid(row=0, column=1, sticky="e")

        ttk.Label(
            input_frame,
            textvariable=self.input_status_var,
            wraplength=360,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        input_frame.columnconfigure(0, weight=1)

        # Shutter speed --------------------------------------------------
        speed_frame = ttk.LabelFrame(sidebar, text="Test shutter speed", padding=5)
        speed_frame.pack(fill="x", pady=(3, 0))

        ttk.Label(
            speed_frame,
            text="Select the marked shutter speed:",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 4))

        for i, speed in enumerate(COMMON_SPEEDS):
            row = 1 + i // 4
            col = i % 4

            button = tk.Radiobutton(
                speed_frame,
                text=speed,
                variable=self.nominal_var,
                value=speed,
                indicatoron=False,
                width=7,
                padx=1,
                pady=2,
                relief="raised",
                offrelief="raised",
                overrelief="ridge",
                command=self._nominal_speed_changed,
            )
            button.grid(row=row, column=col, padx=2, pady=1, sticky="ew")
            speed_frame.columnconfigure(col, weight=1)

        # Exposure tolerance -------------------------------------------
        exposure_settings = ttk.LabelFrame(
            sidebar, text="Exposure tolerance", padding=5
        )
        exposure_settings.pack(fill="x", pady=(3, 0))

        ttk.Label(exposure_settings, text="Allowed error:").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=3
        )
        tolerance = ttk.Combobox(
            exposure_settings,
            textvariable=self.tolerance_var,
            values=list(TOLERANCE_OPTIONS.keys()),
            width=12,
            state="readonly",
        )
        tolerance.grid(row=0, column=1, sticky="ew", pady=2)
        tolerance.bind(
            "<<ComboboxSelected>>", lambda _event: self._refresh_tolerance()
        )
        exposure_settings.columnconfigure(1, weight=1)

        # Expected curtain travel --------------------------------------
        travel_settings = ttk.LabelFrame(
            sidebar, text="Expected curtain travel", padding=5
        )
        travel_settings.pack(fill="x", pady=(3, 0))

        ttk.Label(travel_settings, text="Full-gate time:").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=3
        )
        expected_travel = ttk.Combobox(
            travel_settings,
            textvariable=self.expected_travel_var,
            values=[f"{value:g}" for value in TRAVEL_TIME_PRESETS_MS],
            width=8,
        )
        expected_travel.grid(row=0, column=1, sticky="ew", pady=2)
        expected_travel.bind("<<ComboboxSelected>>", self._expected_travel_changed)
        expected_travel.bind("<FocusOut>", self._expected_travel_changed)
        expected_travel.bind("<Return>", self._expected_travel_changed)

        ttk.Label(travel_settings, text="ms").grid(
            row=0, column=2, sticky="w", padx=(4, 0)
        )

        ttk.Label(travel_settings, text="Expected speed:").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=3
        )
        ttk.Label(
            travel_settings,
            textvariable=self.expected_speed_var,
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=1, column=1, columnspan=2, sticky="w", pady=2)

        ttk.Label(travel_settings, text="Speed tolerance:").grid(
            row=2, column=0, sticky="w", padx=(0, 6), pady=3
        )
        travel_tolerance = ttk.Combobox(
            travel_settings,
            textvariable=self.travel_tolerance_var,
            values=list(TRAVEL_TOLERANCE_OPTIONS.keys()),
            width=8,
            state="readonly",
        )
        travel_tolerance.grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=3
        )
        travel_tolerance.bind(
            "<<ComboboxSelected>>", lambda _event: self._refresh_tolerance()
        )

        ttk.Label(
            travel_settings,
            text=f"Full travel is across the {GATE_MM:g} mm gate.",
            wraplength=355,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(5, 0))
        travel_settings.columnconfigure(1, weight=1)
        self._expected_travel_changed()

        # Simulator controls -------------------------------------------
        self.simulator_frame = ttk.LabelFrame(
            sidebar, text="Simulator input", padding=5
        )
        self.simulator_frame.pack(fill="x", pady=(3, 0))

        ttk.Checkbutton(
            self.simulator_frame,
            text="Follow selected shutter speed",
            variable=self.follow_nominal_var,
            command=self._follow_nominal_changed,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Label(self.simulator_frame, text="Actual exposure:").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self.actual_combo = ttk.Combobox(
            self.simulator_frame,
            textvariable=self.actual_var,
            values=COMMON_SPEEDS,
            width=11,
        )
        self.actual_combo.grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Checkbutton(
            self.simulator_frame,
            text="Curtains follow expected travel",
            variable=self.follow_expected_travel_var,
            command=self._follow_expected_travel_changed,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(3, 2))

        ttk.Label(self.simulator_frame, text="1st curtain:").grid(
            row=3, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self.sim_first_travel_entry = ttk.Entry(
            self.simulator_frame,
            textvariable=self.sim_first_travel_var,
            width=9,
        )
        self.sim_first_travel_entry.grid(row=3, column=1, sticky="ew", pady=2)

        ttk.Label(self.simulator_frame, text="2nd curtain:").grid(
            row=4, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self.sim_second_travel_entry = ttk.Entry(
            self.simulator_frame,
            textvariable=self.sim_second_travel_var,
            width=9,
        )
        self.sim_second_travel_entry.grid(row=4, column=1, sticky="ew", pady=2)

        ttk.Label(
            self.simulator_frame,
            text="Curtain values are full-gate times in ms.",
            wraplength=355,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.simulator_frame.columnconfigure(1, weight=1)

        # Main action buttons ------------------------------------------
        # These remain fixed at the bottom of the sidebar even if the
        # settings pane above is scrolled.
        self.test_button = ttk.Button(
            sidebar_footer,
            text="TEST SIMULATED SHUTTER",
            command=self.run_test,
        )
        self.test_button.grid(row=0, column=0, sticky="ew", ipady=4)

        self.secondary_button = ttk.Button(
            sidebar_footer,
            text="Repeat with new noise",
            command=self.run_test,
        )
        self.secondary_button.grid(
            row=1, column=0, sticky="ew", pady=(4, 0), ipady=1
        )

        self._follow_nominal_changed()
        self._follow_expected_travel_changed()

        # Ensure the settings pane starts at the top after its requested
        # geometry has settled.
        self.root.after_idle(lambda: sidebar_canvas.yview_moveto(0.0))

        # Right: overall result ----------------------------------------
        verdict = ttk.LabelFrame(main, text="Overall tolerance result", padding=10)
        verdict.pack(fill="x")

        self.indicator_canvas = tk.Canvas(
            verdict,
            width=40,
            height=40,
            highlightthickness=0,
        )
        self.indicator_canvas.grid(row=0, column=0, rowspan=2, padx=(2, 12), pady=2)
        self.indicator_circle = self.indicator_canvas.create_oval(
            5, 5, 35, 35,
            fill="#808080",
            outline="",
        )

        self.pass_fail_label = tk.Label(
            verdict,
            textvariable=self.pass_fail_var,
            font=("TkDefaultFont", 16, "bold"),
            anchor="w",
        )
        self.pass_fail_label.grid(row=0, column=1, sticky="w")

        ttk.Label(
            verdict,
            textvariable=self.tolerance_detail_var,
            wraplength=760,
        ).grid(row=1, column=1, columnspan=3, sticky="w", pady=(2, 0))

        ttk.Label(
            verdict,
            text="Variance from selected speed:",
        ).grid(row=0, column=2, sticky="e", padx=(35, 6))

        ttk.Label(
            verdict,
            textvariable=self.variance_var,
            font=("TkDefaultFont", 13, "bold"),
        ).grid(row=0, column=3, sticky="w")

        verdict.columnconfigure(1, weight=1)

        # Curtain result ------------------------------------------------
        travel_verdict = ttk.LabelFrame(
            main, text="Curtain travel measurements", padding=10
        )
        travel_verdict.pack(fill="x", pady=(10, 0))

        self.travel1_indicator_canvas = tk.Canvas(
            travel_verdict, width=30, height=30, highlightthickness=0
        )
        self.travel1_indicator_canvas.grid(row=0, column=0, rowspan=2, padx=(2, 8))
        self.travel1_indicator_circle = self.travel1_indicator_canvas.create_oval(
            4, 4, 26, 26, fill="#808080", outline=""
        )
        self.travel1_status_label = tk.Label(
            travel_verdict,
            textvariable=self.travel1_status_var,
            font=("TkDefaultFont", 11, "bold"),
            anchor="w",
        )
        self.travel1_status_label.grid(row=0, column=1, sticky="w")
        ttk.Label(travel_verdict, text="1st curtain:").grid(
            row=1, column=1, sticky="w", padx=(0, 5)
        )
        ttk.Label(
            travel_verdict,
            textvariable=self.travel1_var,
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=1, column=2, sticky="w", padx=(0, 30))

        self.travel2_indicator_canvas = tk.Canvas(
            travel_verdict, width=30, height=30, highlightthickness=0
        )
        self.travel2_indicator_canvas.grid(row=0, column=3, rowspan=2, padx=(2, 8))
        self.travel2_indicator_circle = self.travel2_indicator_canvas.create_oval(
            4, 4, 26, 26, fill="#808080", outline=""
        )
        self.travel2_status_label = tk.Label(
            travel_verdict,
            textvariable=self.travel2_status_var,
            font=("TkDefaultFont", 11, "bold"),
            anchor="w",
        )
        self.travel2_status_label.grid(row=0, column=4, sticky="w")
        ttk.Label(travel_verdict, text="2nd curtain:").grid(
            row=1, column=4, sticky="w", padx=(0, 5)
        )
        ttk.Label(
            travel_verdict,
            textvariable=self.travel2_var,
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=1, column=5, sticky="w")

        ttk.Label(
            travel_verdict,
            textvariable=self.travel_detail_var,
            wraplength=900,
        ).grid(row=2, column=0, columnspan=6, sticky="w", pady=(7, 0))

        travel_verdict.columnconfigure(2, weight=1)
        travel_verdict.columnconfigure(5, weight=1)

        # Detailed values ----------------------------------------------
        results = ttk.LabelFrame(main, text="Measurement", padding=10)
        results.pack(fill="x", pady=(10, 0))

        fields = [
            ("Method", self.method_var),
            ("Average exposure", self.avg_var),
            ("Sensor 1 exposure", self.s1_var),
            ("Sensor 2 exposure", self.s2_var),
            ("Sensor 1 variance", self.s1_variance_var),
            ("Sensor 2 variance", self.s2_variance_var),
            ("Exposure mismatch", self.mismatch_var),
            ("Confidence", self.conf_var),
        ]

        columns_per_row = 3
        for i, (label, var) in enumerate(fields):
            row = i // columns_per_row
            pair = i % columns_per_row
            col = pair * 2

            ttk.Label(results, text=label + ":").grid(
                row=row, column=col, sticky="w", padx=(0, 5), pady=4
            )
            ttk.Label(
                results,
                textvariable=var,
                font=("TkDefaultFont", 10, "bold"),
            ).grid(
                row=row, column=col + 1, sticky="w", padx=(0, 24), pady=4
            )

        # Plot ----------------------------------------------------------
        plot_frame = ttk.LabelFrame(main, text="Stereo line-in waveform", padding=6)
        plot_frame.pack(fill="both", expand=True, pady=(10, 0))

        self.figure = Figure(figsize=(9, 4.8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.25)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        status = ttk.Label(
            outer,
            textvariable=self.status_var,
            anchor="w",
            relief="sunken",
        )
        status.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    # ------------------------------------------------------------------
    # Device enumeration / continuous audio
    # ------------------------------------------------------------------

    def refresh_input_devices(self):
        selected = self.input_var.get()
        self.device_map = {"Simulator": None}
        labels = ["Simulator"]

        if sd is not None:
            try:
                devices = sd.query_devices()
                for idx, dev in enumerate(devices):
                    max_in = int(dev["max_input_channels"])
                    if max_in <= 0:
                        continue

                    name = str(dev["name"])
                    default_rate = int(round(float(dev["default_samplerate"])))
                    capability = "stereo" if max_in >= 2 else "mono"
                    label = (
                        f"{idx}: {name} [{capability}, "
                        f"default {default_rate} Hz]"
                    )
                    self.device_map[label] = idx
                    labels.append(label)
            except Exception as exc:
                self.input_status_var.set(f"Could not list audio devices: {exc}")
        else:
            self.input_status_var.set(
                "sounddevice is not installed; simulator only. "
                "Install with: pip install sounddevice"
            )

        self.input_combo["values"] = labels

        if selected in self.device_map:
            self.input_var.set(selected)
        else:
            self.input_var.set("Simulator")
            self._stop_audio_stream()

    def _input_changed(self, _event=None):
        self.cancel_live_test(silent=True)
        selected = self.input_var.get()
        device_index = self.device_map.get(selected)

        if device_index is None:
            self._stop_audio_stream()
            self.input_status_var.set("Simulator selected")
            self.status_var.set("Ready — simulator input")
            self.test_button.configure(
                text="TEST SIMULATED SHUTTER",
                command=self.run_test,
            )
            self.secondary_button.configure(
                text="Repeat with new noise",
                command=self.run_test,
            )
            return

        try:
            self._start_audio_stream(device_index)
            self.test_button.configure(
                text="ARM LIVE INPUT",
                command=self.arm_live_test,
            )
            self.secondary_button.configure(
                text="CANCEL LIVE TEST",
                command=self.cancel_live_test,
            )
        except Exception as exc:
            self.input_status_var.set(f"Audio input error: {exc}")
            messagebox.showerror("Audio input", str(exc))

    def _choose_input_sample_rate(self, device_index: int) -> int:
        if sd is None:
            raise RuntimeError("sounddevice is not installed.")

        dev = sd.query_devices(device_index)
        if int(dev["max_input_channels"]) < 2:
            raise RuntimeError(
                "The selected device has fewer than 2 input channels. "
                "It is listed in the selector, but two sensor channels are "
                "required for shutter and curtain-travel timing."
            )

        # First choice: exactly 96 kHz.
        try:
            sd.check_input_settings(
                device=device_index,
                channels=2,
                samplerate=PREFERRED_FS,
                dtype="float32",
            )
            return PREFERRED_FS
        except Exception:
            pass

        # Fallback: use the device's reported default rate, but always feed
        # that ACTUAL rate into timing calculations.
        default_fs = int(round(float(dev["default_samplerate"])))
        sd.check_input_settings(
            device=device_index,
            channels=2,
            samplerate=default_fs,
            dtype="float32",
        )
        return default_fs

    def _start_audio_stream(self, device_index: int):
        if sd is None:
            raise RuntimeError(
                "Live audio requires sounddevice. Install with: pip install sounddevice"
            )

        self._stop_audio_stream()

        fs = self._choose_input_sample_rate(device_index)
        dev = sd.query_devices(device_index)
        capacity = int(math.ceil(LIVE_BUFFER_SECONDS * fs))

        self.audio_buffer = StereoRingBuffer(capacity)
        self.audio_fs = fs
        self.audio_device_index = device_index
        self.audio_stream_error = None

        def callback(indata, frames, time_info, status):
            del frames, time_info
            if status:
                self.audio_stream_error = str(status)
            if self.audio_buffer is not None:
                self.audio_buffer.write(indata)

        self.audio_stream = sd.InputStream(
            device=device_index,
            channels=2,
            samplerate=fs,
            dtype="float32",
            callback=callback,
            blocksize=0,
        )
        self.audio_stream.start()

        self.input_status_var.set(
            f"Streaming continuously: {dev['name']} — 2 ch @ {fs:,} Hz"
        )
        self.status_var.set(
            "Live stream running — select shutter speed and press ARM LIVE INPUT"
        )

    def _stop_audio_stream(self):
        stream = self.audio_stream
        self.audio_stream = None
        self.audio_buffer = None
        self.audio_fs = None
        self.audio_device_index = None

        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Live capture state machine
    # ------------------------------------------------------------------

    def arm_live_test(self):
        try:
            if self.audio_stream is None or self.audio_buffer is None or self.audio_fs is None:
                raise RuntimeError("Select a live audio input first.")

            nominal_s = parse_speed(self.nominal_var.get())
            expected_ms = self._parse_expected_travel_ms()
            fs = self.audio_fs

            baseline_frames = int(LIVE_BASELINE_SECONDS * fs)
            baseline = self.audio_buffer.last(baseline_frames)

            # A little baseline is required so noise does not become the trigger.
            if len(baseline) < max(256, int(0.030 * fs)):
                raise RuntimeError(
                    "Not enough live baseline samples yet. The audio stream must "
                    "contain at least about 30 ms before arming."
                )

            med = np.median(baseline, axis=0)
            mad = np.median(np.abs(baseline - med), axis=0)
            sigma = np.maximum(1e-7, 1.4826 * mad)
            threshold = np.maximum(LIVE_MIN_TRIGGER, LIVE_TRIGGER_SIGMA * sigma)

            _, newest = self.audio_buffer.bounds()

            self.live_nominal_s = nominal_s
            self.live_expected_travel_ms = expected_ms
            self.live_transient_mode = uses_transient_method(nominal_s, expected_ms)
            self.live_baseline = med.astype(np.float64)
            self.live_threshold = threshold.astype(np.float64)
            self.live_scan_index = newest

            self.live_first_open_index = None
            self.live_deadline_index = None
            self.live_capture_end_index = None
            self.live_open_seen = [False, False]
            self.live_close_seen = [False, False]
            self.live_open_indices = [None, None]
            self.live_close_indices = [None, None]

            self.live_state = "armed"
            self.result = None
            self._clear_result_values()
            self._set_indicator(None)
            self._set_small_indicator(
                self.travel1_indicator_canvas,
                self.travel1_indicator_circle,
                self.travel1_status_var,
                self.travel1_status_label,
                None,
            )
            self._set_small_indicator(
                self.travel2_indicator_canvas,
                self.travel2_indicator_circle,
                self.travel2_status_var,
                self.travel2_status_label,
                None,
            )

            mode_text = (
                "edge mode: waiting for opening then both closing transients"
                if self.live_transient_mode
                else "fast/template mode: waiting for shutter pulse"
            )
            self.status_var.set(
                f"ARMED — {mode_text}. Trigger thresholds: "
                f"S1 {threshold[0]:.4f}, S2 {threshold[1]:.4f}"
            )
            self.tolerance_detail_var.set("Armed — waiting for the next shutter event.")

        except Exception as exc:
            messagebox.showerror("Live shutter test", str(exc))

    def cancel_live_test(self, silent=False):
        if self.live_state != "idle":
            self.live_state = "idle"
            if not silent:
                self.status_var.set("Live test cancelled — audio stream is still running")
                self.tolerance_detail_var.set(
                    "Live capture cancelled. Press ARM LIVE INPUT for another test."
                )

    def _poll_live_input(self):
        try:
            if self.audio_stream_error:
                err = self.audio_stream_error
                self.audio_stream_error = None
                self.input_status_var.set(f"Audio stream warning: {err}")

            if (
                self.live_state != "idle"
                and self.audio_buffer is not None
                and self.audio_fs is not None
            ):
                self._process_live_samples()

        except Exception as exc:
            self.live_state = "idle"
            self.status_var.set(f"Live input error: {exc}")
            self._show_incomplete(
                f"Live capture error: {exc}",
                plot_partial=False,
            )
        finally:
            self.root.after(LIVE_POLL_MS, self._poll_live_input)

    def _process_live_samples(self):
        assert self.audio_buffer is not None
        assert self.audio_fs is not None

        oldest, newest = self.audio_buffer.bounds()
        if newest <= self.live_scan_index:
            return

        # If the ring buffer has advanced beyond the samples we still need,
        # abort rather than analyzing the wrong event.
        if self.live_scan_index < oldest:
            self.live_state = "idle"
            self._show_incomplete(
                "Live ring buffer overrun before the event could be completed.",
                plot_partial=False,
            )
            return

        scan_start = self.live_scan_index
        chunk = self.audio_buffer.get(scan_start, newest)
        self.live_scan_index = newest

        centered = chunk.astype(np.float64) - self.live_baseline[None, :]

        # Detect positive opening features independently on each channel.
        for c in range(2):
            if not self.live_open_seen[c]:
                hits = np.flatnonzero(centered[:, c] >= self.live_threshold[c])
                if len(hits):
                    idx = scan_start + int(hits[0])
                    self.live_open_seen[c] = True
                    self.live_open_indices[c] = idx

                    if self.live_first_open_index is None or idx < self.live_first_open_index:
                        self.live_first_open_index = idx

        # First opening transition starts the measurement deadline.
        if self.live_state == "armed" and self.live_first_open_index is not None:
            first_open = self.live_first_open_index
            fs = self.audio_fs

            if self.live_transient_mode:
                timeout_s = live_close_timeout_s(
                    self.live_nominal_s,
                    self.live_expected_travel_ms,
                )
                self.live_deadline_index = first_open + int(math.ceil(timeout_s * fs))
                self.live_state = "waiting_close"
                slow_stops = live_timeout_slow_stops(self.live_nominal_s)
                self.status_var.set(
                    f"OPEN DETECTED — waiting for close. "
                    f"Timeout {timeout_s * 1000:.1f} ms "
                    f"(allows +{slow_stops:.0f} stops slow)."
                )
            else:
                capture_s = max(
                    LIVE_FAST_CAPTURE_SECONDS,
                    4.0 * self.live_nominal_s
                    + 2.0 * self.live_expected_travel_ms / 1000.0
                    + 0.015,
                )
                self.live_capture_end_index = first_open + int(math.ceil(capture_s * fs))
                self.live_deadline_index = self.live_capture_end_index
                self.live_state = "fast_capture"
                self.status_var.set(
                    f"FAST SHUTTER PULSE DETECTED — collecting "
                    f"{capture_s * 1000:.1f} ms for template analysis."
                )

        # For edge-mode exposures, closing transients are negative excursions.
        if self.live_state == "waiting_close":
            # Don't accept a negative sample occurring before that channel opened.
            for c in range(2):
                if self.live_open_seen[c] and not self.live_close_seen[c]:
                    opened_at = self.live_open_indices[c]
                    if opened_at is None:
                        continue

                    min_rel = max(0, opened_at - scan_start + 1)
                    if min_rel >= len(centered):
                        continue

                    tail = centered[min_rel:, c]
                    hits = np.flatnonzero(tail <= -self.live_threshold[c])
                    if len(hits):
                        idx = scan_start + min_rel + int(hits[0])
                        self.live_close_seen[c] = True
                        self.live_close_indices[c] = idx

            if all(self.live_open_seen) and all(self.live_close_seen):
                last_close = max(int(i) for i in self.live_close_indices if i is not None)
                self.live_capture_end_index = (
                    last_close + int(LIVE_POSTROLL_SECONDS * self.audio_fs)
                )
                self.live_state = "postroll"
                self.status_var.set(
                    "Both closing transients detected — collecting short post-roll…"
                )

            elif (
                self.live_deadline_index is not None
                and newest >= self.live_deadline_index
            ):
                self._live_timeout()
                return

        if self.live_state in ("postroll", "fast_capture"):
            if (
                self.live_capture_end_index is not None
                and newest >= self.live_capture_end_index
            ):
                self._finish_live_capture()

    def _live_timeout(self):
        assert self.audio_buffer is not None
        assert self.audio_fs is not None
        assert self.live_first_open_index is not None
        assert self.live_deadline_index is not None

        missing = []
        for c in range(2):
            if not self.live_open_seen[c]:
                missing.append(f"sensor {c + 1} opening")
            elif not self.live_close_seen[c]:
                missing.append(f"sensor {c + 1} closing")

        start = max(
            self.audio_buffer.bounds()[0],
            self.live_first_open_index
            - int(LIVE_PRETRIGGER_SECONDS * self.audio_fs),
        )
        end = min(self.audio_buffer.bounds()[1], self.live_deadline_index)

        try:
            partial = self.audio_buffer.get(start, end)
            self.samples = partial
            self.sample_rate = self.audio_fs
            self.plot_source_label = "live input — incomplete"
            self._update_partial_plot(
                partial,
                self.audio_fs,
                note="INCOMPLETE / TIMEOUT",
            )
        except Exception:
            partial = None

        timeout_s = live_close_timeout_s(
            self.live_nominal_s,
            self.live_expected_travel_ms,
        )
        slow_stops = live_timeout_slow_stops(self.live_nominal_s)
        missing_text = ", ".join(missing) if missing else "valid closing event"
        self.live_state = "idle"

        self._show_incomplete(
            f"Opening was detected, but {missing_text} was not detected within "
            f"{timeout_s * 1000:.1f} ms "
            f"(+{slow_stops:.0f}-stop slow allowance). Measurement discarded.",
            plot_partial=False,
        )

    def _finish_live_capture(self):
        assert self.audio_buffer is not None
        assert self.audio_fs is not None
        assert self.live_first_open_index is not None
        assert self.live_capture_end_index is not None

        fs = self.audio_fs
        oldest, newest = self.audio_buffer.bounds()
        start = max(
            oldest,
            self.live_first_open_index - int(LIVE_PRETRIGGER_SECONDS * fs),
        )
        end = min(newest, self.live_capture_end_index)

        samples = self.audio_buffer.get(start, end)
        self.live_state = "idle"

        try:
            result = analyze_signal(
                samples,
                self.live_nominal_s,
                fs=fs,
                expected_travel_ms=self.live_expected_travel_ms,
            )

            self.samples = samples
            self.sample_rate = fs
            self.result = result
            self.meta = None
            self.plot_source_label = "live input"

            self._update_results(result, self.live_nominal_s)
            self._update_plot(samples, result, fs=fs, source_label="live input")

            measured_text = format_speed(result.exposure_average_s)
            error_stops = stops_from_nominal(
                result.exposure_average_s,
                self.live_nominal_s,
            )
            self.status_var.set(
                f"Done — live input measured {measured_text}, "
                f"error {error_stops:+.2f} stops, method: {result.method}, "
                f"{fs:,} Hz"
            )

        except Exception as exc:
            self._show_incomplete(
                f"Shutter event was captured, but analysis could not produce "
                f"a valid measurement: {exc}",
                plot_partial=False,
            )
            self._update_partial_plot(
                samples,
                fs,
                note="Captured event — analysis failed",
            )

    # ------------------------------------------------------------------
    # Settings helpers
    # ------------------------------------------------------------------

    def _nominal_speed_changed(self):
        if self.follow_nominal_var.get():
            self.actual_var.set(self.nominal_var.get())

    def _follow_nominal_changed(self):
        follows = self.follow_nominal_var.get()
        if follows:
            self.actual_var.set(self.nominal_var.get())
            self.actual_combo.configure(state="disabled")
        else:
            self.actual_combo.configure(state="normal")

    def _parse_expected_travel_ms(self) -> float:
        value = float(self.expected_travel_var.get().strip())
        if value <= 0:
            raise ValueError("Expected curtain travel time must be positive.")
        return value

    @staticmethod
    def _parse_positive_ms(value: str, label: str) -> float:
        parsed = float(value.strip())
        if parsed <= 0:
            raise ValueError(f"{label} must be positive.")
        return parsed

    def _expected_travel_changed(self, _event=None):
        try:
            expected_ms = self._parse_expected_travel_ms()
        except ValueError:
            self.expected_speed_var.set("invalid value")
            return

        speed = curtain_speed_m_s(expected_ms / 1000.0)
        self.expected_speed_var.set(f"{speed:.2f} m/s")

        if self.follow_expected_travel_var.get():
            text = f"{expected_ms:g}"
            self.sim_first_travel_var.set(text)
            self.sim_second_travel_var.set(text)

        if self.result is not None:
            self._refresh_tolerance()

    def _follow_expected_travel_changed(self):
        follows = self.follow_expected_travel_var.get()
        state = "disabled" if follows else "normal"
        self.sim_first_travel_entry.configure(state=state)
        self.sim_second_travel_entry.configure(state=state)
        if follows:
            self._expected_travel_changed()

    def _get_tolerance_stops(self) -> float:
        return TOLERANCE_OPTIONS.get(self.tolerance_var.get(), 1.0 / 3.0)

    def _get_travel_tolerance_percent(self) -> float:
        return TRAVEL_TOLERANCE_OPTIONS.get(
            self.travel_tolerance_var.get(), 10.0
        )

    def _refresh_tolerance(self):
        if self.result is not None:
            nominal_s = parse_speed(self.nominal_var.get())
            self._update_results(self.result, nominal_s)

    # ------------------------------------------------------------------
    # Result state
    # ------------------------------------------------------------------

    def _clear_result_values(self):
        self.method_var.set("—")
        self.avg_var.set("—")
        self.s1_var.set("—")
        self.s2_var.set("—")
        self.variance_var.set("—")
        self.s1_variance_var.set("—")
        self.s2_variance_var.set("—")
        self.travel1_var.set("—")
        self.travel2_var.set("—")
        self.mismatch_var.set("—")
        self.conf_var.set("—")

    def _set_indicator(self, passed: bool | None):
        if passed is None:
            color = "#808080"
            text = "NOT TESTED"
            text_color = "#404040"
        elif passed:
            color = "#2e9b50"
            text = "PASS"
            text_color = "#20743b"
        else:
            color = "#d33f3f"
            text = "FAIL"
            text_color = "#a32727"

        self.indicator_canvas.itemconfigure(self.indicator_circle, fill=color)
        self.pass_fail_var.set(text)
        self.pass_fail_label.configure(fg=text_color)

    def _set_incomplete_indicator(self):
        self.indicator_canvas.itemconfigure(
            self.indicator_circle,
            fill="#d08a1f",
        )
        self.pass_fail_var.set("INCOMPLETE")
        self.pass_fail_label.configure(fg="#9b650f")

    @staticmethod
    def _set_small_indicator(
        canvas: tk.Canvas,
        circle: int,
        status_var: tk.StringVar,
        label: tk.Label,
        passed: bool | None,
    ):
        if passed is None:
            color, text, text_color = "#808080", "NOT MEASURED", "#404040"
        elif passed:
            color, text, text_color = "#2e9b50", "PASS", "#20743b"
        else:
            color, text, text_color = "#d33f3f", "FAIL", "#a32727"

        canvas.itemconfigure(circle, fill=color)
        status_var.set(text)
        label.configure(fg=text_color)

    def _show_incomplete(self, detail: str, plot_partial=False):
        del plot_partial
        self.result = None
        self._clear_result_values()
        self._set_incomplete_indicator()

        self._set_small_indicator(
            self.travel1_indicator_canvas,
            self.travel1_indicator_circle,
            self.travel1_status_var,
            self.travel1_status_label,
            None,
        )
        self._set_small_indicator(
            self.travel2_indicator_canvas,
            self.travel2_indicator_circle,
            self.travel2_status_var,
            self.travel2_status_label,
            None,
        )

        self.tolerance_detail_var.set(detail)
        self.travel_detail_var.set(
            "Curtain travel was not judged because the shutter event was incomplete."
        )
        self.status_var.set("INCOMPLETE / TIMEOUT — " + detail)

    # ------------------------------------------------------------------
    # Simulator test
    # ------------------------------------------------------------------

    def run_test(self):
        # If a live input is selected, the same large button arms it.
        if self.device_map.get(self.input_var.get()) is not None:
            self.arm_live_test()
            return

        try:
            nominal_s = parse_speed(self.nominal_var.get())
            expected_travel_ms = self._parse_expected_travel_ms()

            if self.follow_nominal_var.get():
                self.actual_var.set(self.nominal_var.get())

            actual_s = parse_speed(self.actual_var.get())

            if self.follow_expected_travel_var.get():
                actual_first_travel_ms = expected_travel_ms
                actual_second_travel_ms = expected_travel_ms
                text = f"{expected_travel_ms:g}"
                self.sim_first_travel_var.set(text)
                self.sim_second_travel_var.set(text)
            else:
                actual_first_travel_ms = self._parse_positive_ms(
                    self.sim_first_travel_var.get(),
                    "Simulated 1st-curtain travel",
                )
                actual_second_travel_ms = self._parse_positive_ms(
                    self.sim_second_travel_var.get(),
                    "Simulated 2nd-curtain travel",
                )

            self.status_var.set("Generating simulated stereo signal…")
            self.root.update_idletasks()

            samples, meta = simulate_shutter(
                actual_s,
                fs=PREFERRED_FS,
                travel_ms=actual_first_travel_ms,
                second_travel_ms=actual_second_travel_ms,
            )

            self.status_var.set("Analyzing…")
            self.root.update_idletasks()

            result = analyze_signal(
                samples,
                nominal_s,
                fs=PREFERRED_FS,
                expected_travel_ms=expected_travel_ms,
            )

            self.samples = samples
            self.sample_rate = PREFERRED_FS
            self.meta = meta
            self.result = result
            self.plot_source_label = "simulated AC-coupled input"

            self._update_results(result, nominal_s)
            self._update_plot(
                samples,
                result,
                fs=PREFERRED_FS,
                source_label="simulated AC-coupled input",
            )

            actual_text = format_speed(actual_s)
            measured_text = format_speed(result.exposure_average_s)
            error_stops = stops_from_nominal(
                result.exposure_average_s,
                nominal_s,
            )

            self.status_var.set(
                f"Done — simulated {actual_text}, measured {measured_text}, "
                f"error {error_stops:+.2f} stops, method: {result.method}"
            )

        except Exception as exc:
            self.status_var.set("Error")
            self._set_indicator(None)
            self._set_small_indicator(
                self.travel1_indicator_canvas,
                self.travel1_indicator_circle,
                self.travel1_status_var,
                self.travel1_status_label,
                None,
            )
            self._set_small_indicator(
                self.travel2_indicator_canvas,
                self.travel2_indicator_circle,
                self.travel2_status_var,
                self.travel2_status_label,
                None,
            )
            messagebox.showerror("Shutter tester", str(exc))

    # ------------------------------------------------------------------
    # Result calculations / plot
    # ------------------------------------------------------------------

    def _update_results(self, r: AnalysisResult, nominal_s: float):
        self.method_var.set(r.method)
        self.avg_var.set(format_speed(r.exposure_average_s))
        self.s1_var.set(format_speed(r.exposure_sensor_1_s))
        self.s2_var.set(format_speed(r.exposure_sensor_2_s))

        avg_stops = stops_from_nominal(r.exposure_average_s, nominal_s)
        s1_stops = stops_from_nominal(r.exposure_sensor_1_s, nominal_s)
        s2_stops = stops_from_nominal(r.exposure_sensor_2_s, nominal_s)

        self.variance_var.set(format_stop_error(avg_stops))
        self.s1_variance_var.set(f"{s1_stops:+.2f} stops")
        self.s2_variance_var.set(f"{s2_stops:+.2f} stops")

        expected_travel_s = self._parse_expected_travel_ms() / 1000.0
        travel1_error = travel_error_percent(
            r.first_curtain_full_travel_s,
            expected_travel_s,
        )
        travel2_error = travel_error_percent(
            r.second_curtain_full_travel_s,
            expected_travel_s,
        )

        self.travel1_var.set(
            format_travel_measurement(
                r.first_curtain_full_travel_s,
                expected_travel_s,
            )
        )
        self.travel2_var.set(
            format_travel_measurement(
                r.second_curtain_full_travel_s,
                expected_travel_s,
            )
        )

        self.mismatch_var.set(f"{r.sensor_mismatch_percent:.2f}%")
        self.conf_var.set(f"{r.confidence * 100:.0f}%")

        tolerance = self._get_tolerance_stops()
        travel_tolerance = self._get_travel_tolerance_percent()

        exposure_passed = (
            np.isfinite(s1_stops)
            and np.isfinite(s2_stops)
            and abs(s1_stops) <= tolerance
            and abs(s2_stops) <= tolerance
        )

        travel1_passed = (
            np.isfinite(travel1_error)
            and abs(travel1_error) <= travel_tolerance
        )
        travel2_passed = (
            np.isfinite(travel2_error)
            and abs(travel2_error) <= travel_tolerance
        )

        self._set_small_indicator(
            self.travel1_indicator_canvas,
            self.travel1_indicator_circle,
            self.travel1_status_var,
            self.travel1_status_label,
            travel1_passed,
        )
        self._set_small_indicator(
            self.travel2_indicator_canvas,
            self.travel2_indicator_circle,
            self.travel2_status_var,
            self.travel2_status_label,
            travel2_passed,
        )

        self.travel_detail_var.set(
            f"Expected {expected_travel_s * 1000:.3f} ms "
            f"({curtain_speed_m_s(expected_travel_s):.2f} m/s); "
            f"allowed curtain-speed error {self.travel_tolerance_var.get()}."
        )

        passed = exposure_passed and travel1_passed and travel2_passed
        self._set_indicator(passed)

        worst = max(abs(s1_stops), abs(s2_stops))
        if passed:
            detail = (
                f"Exposure and both curtains are within tolerance. "
                f"Worst exposure side: {worst:.2f} stops."
            )
        else:
            failed_parts = []
            if not exposure_passed:
                failed_parts.append("exposure")
            if not travel1_passed:
                failed_parts.append("1st-curtain travel")
            if not travel2_passed:
                failed_parts.append("2nd-curtain travel")
            detail = "Outside tolerance: " + ", ".join(failed_parts) + "."

        self.tolerance_detail_var.set(detail)

    def _update_plot(
        self,
        samples: np.ndarray,
        r: AnalysisResult,
        fs: int,
        source_label: str,
    ):
        self.ax.clear()

        t_ms = np.arange(len(samples)) / fs * 1000.0
        self.ax.plot(t_ms, samples[:, 0], label="Sensor 1", linewidth=1.0)
        self.ax.plot(t_ms, samples[:, 1], label="Sensor 2", linewidth=1.0)

        markers = [
            ("sensor1_open", "S1 open"),
            ("sensor1_close", "S1 close"),
            ("sensor2_open", "S2 open"),
            ("sensor2_close", "S2 close"),
        ]

        for key, _label in markers:
            idx = r.event_indices.get(key)
            if idx is not None and 0 <= idx < len(samples):
                self.ax.axvline(
                    idx / fs * 1000.0,
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.55,
                )

        active = list(r.event_indices.values())
        if active:
            lo = max(0, min(active) - int(0.004 * fs))
            hi = min(len(samples) - 1, max(active) + int(0.008 * fs))
            if hi > lo:
                self.ax.set_xlim(t_ms[lo], t_ms[hi])

        self.ax.set_title(
            f"{fs / 1000:.1f} kHz {source_label} — analyzer: {r.method}"
        )
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Line-in amplitude")
        self.ax.grid(True, alpha=0.25)
        self.ax.legend(loc="upper right")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _update_partial_plot(self, samples: np.ndarray, fs: int, note: str):
        self.ax.clear()
        t_ms = np.arange(len(samples)) / fs * 1000.0
        self.ax.plot(t_ms, samples[:, 0], label="Sensor 1", linewidth=1.0)
        self.ax.plot(t_ms, samples[:, 1], label="Sensor 2", linewidth=1.0)
        self.ax.set_title(f"{fs / 1000:.1f} kHz live input — {note}")
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Line-in amplitude")
        self.ax.grid(True, alpha=0.25)
        self.ax.legend(loc="upper right")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _on_close(self):
        self.live_state = "idle"
        self._stop_audio_stream()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    # Simulator/analyzer sanity.
    for speed in ("1/30", "1/250", "1/1000", "1/4000"):
        nominal = parse_speed(speed)
        samples, _ = simulate_shutter(
            nominal,
            fs=PREFERRED_FS,
            noise_rms=0.0,
        )
        r = analyze_signal(
            samples,
            nominal,
            fs=PREFERRED_FS,
            expected_travel_ms=CURTAIN_TRAVEL_MS,
        )
        assert np.isfinite(r.exposure_average_s), speed
        assert r.exposure_average_s > 0, speed

    # Ring-buffer wrap sanity.
    rb = StereoRingBuffer(100)
    a = np.column_stack((np.arange(70), np.arange(70))).astype(np.float32)
    b = np.column_stack((np.arange(70, 140), np.arange(70, 140))).astype(np.float32)
    rb.write(a)
    rb.write(b)
    oldest, newest = rb.bounds()
    assert (oldest, newest) == (40, 140)
    last = rb.get(120, 140)
    assert np.allclose(last[:, 0], np.arange(120, 140))

    # Timeout bands for initial tensioning.
    assert live_timeout_slow_stops(parse_speed("1/30")) == 2.0
    assert live_timeout_slow_stops(parse_speed("1/60")) == 3.0
    assert live_timeout_slow_stops(parse_speed("1/1000")) == 3.0
    assert live_timeout_slow_stops(parse_speed("1/2000")) == 4.0
    assert live_timeout_slow_stops(parse_speed("1/8000")) == 4.0

    t_30 = live_close_timeout_s(parse_speed("1/30"), CURTAIN_TRAVEL_MS)
    t_60 = live_close_timeout_s(parse_speed("1/60"), CURTAIN_TRAVEL_MS)
    t_1000 = live_close_timeout_s(parse_speed("1/1000"), CURTAIN_TRAVEL_MS)
    t_2000 = live_close_timeout_s(parse_speed("1/2000"), CURTAIN_TRAVEL_MS)

    assert t_30 > parse_speed("1/30") * 4.0
    assert t_60 > parse_speed("1/60") * 8.0
    assert t_1000 >= 0.050
    assert t_2000 >= 0.050

    print("Self-test passed.")
    print(f"1/30 timeout:   {t_30 * 1000:.1f} ms (+2 stops)")
    print(f"1/60 timeout:   {t_60 * 1000:.1f} ms (+3 stops)")
    print(f"1/1000 timeout: {t_1000 * 1000:.1f} ms (+3 stops)")
    print(f"1/2000 timeout: {t_2000 * 1000:.1f} ms (+4 stops)")


def main():
    if "--self-test" in sys.argv:
        self_test()
        return

    root = tk.Tk()
    ShutterTesterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
