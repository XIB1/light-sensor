#!/usr/bin/env python3
"""
Two-point shutter tester prototype.

Current input source:
    Simulated 96 kHz stereo signal.

Measurements:
    Exposure at both sensors, exposure error in stops, and independent first-
    and second-curtain full-gate travel time/speed with selectable tolerances.

Future input source:
    Real stereo line-in. The analyzer is deliberately separated from the
    simulator so the real input can later provide the same (N, 2) NumPy array.

Dependencies:
    pip install numpy matplotlib

Run:
    python shutter_tester_v2.py
"""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk, messagebox

import numpy as np

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError as exc:
    raise SystemExit("Missing matplotlib. Install with: pip install numpy matplotlib") from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FS = 96_000

# Dimension along curtain travel. Change to 24 mm for many vertical shutters,
# or 36 mm for a horizontal-travel shutter.
GATE_MM = 24.0

# Full gate traversal time. This is only a simulation default and will later
# be measured from the two real sensors.
CURTAIN_TRAVEL_MS = 3.3

# Effective optical footprint of the sensor assembly, not the bare die size.
SENSOR_WIDTH_MM = 2.0

# Approximate high-pass cutoff caused by the AC-coupled line input.
# Calibrate this from a real long-exposure recording later.
AC_COUPLING_HZ = 35.0

# Approximate SFH309 / electronics low-pass response.
# The bare phototransistor is fast; this is intentionally modest.
SENSOR_TAU_US = 7.0

NOISE_RMS = 0.003

COMMON_SPEEDS = [
    "1", "1/2", "1/4", "1/8", "1/15", "1/30", "1/60", "1/125",
    "1/250", "1/500", "1/1000", "1/2000", "1/4000", "1/8000"
]


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
    """
    First-order high-pass approximation for an AC-coupled sound-card input.
    """
    if cutoff_hz <= 0:
        return x.copy()

    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / fs
    alpha = rc / (rc + dt)

    y = np.zeros_like(x, dtype=np.float64)
    for i in range(1, len(x)):
        y[i] = alpha * (y[i - 1] + x[i] - x[i - 1])
    return y


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
    """
    Average illumination over a finite sensor footprint.

    For each position x:
        first curtain opens at x / v1
        second curtain closes at exposure + x / v2

    ``exposure_s`` is the curtain separation in time at x = 0. If the two
    curtains travel at different speeds, exposure therefore changes across
    the gate, as it does with real shutter capping or taper.
    """
    width = x_end_mm - x_start_mm
    dx = width / spatial_samples
    xs = x_start_mm + (np.arange(spatial_samples) + 0.5) * dx

    t_open = xs / first_curtain_speed_mm_s
    t_close = exposure_s + xs / second_curtain_speed_mm_s

    lit = (
        (t[:, None] >= t_open[None, :]) &
        (t[:, None] < t_close[None, :])
    )
    return lit.mean(axis=1, dtype=np.float64)


def simulate_shutter(
    exposure_s: float,
    fs: int = FS,
    gate_mm: float = GATE_MM,
    travel_ms: float = CURTAIN_TRAVEL_MS,
    second_travel_ms: float | None = None,
    sensor_width_mm: float = SENSOR_WIDTH_MM,
    ac_hz: float = AC_COUPLING_HZ,
    sensor_tau_us: float = SENSOR_TAU_US,
    noise_rms: float = NOISE_RMS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Return stereo float32 samples, shape (N, 2).

    CH1 = first sensor
    CH2 = second sensor
    """
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

    # Put sensor footprints just inside the two opposite edges.
    s1 = (0.0, sensor_width_mm)
    s2 = (gate_mm - sensor_width_mm, gate_mm)

    # Include enough pre/post-roll for the AC-coupled transients.
    # For very long exposures, keep the test buffer manageable but complete.
    pre_s = 0.012
    post_s = max(0.030, 5.0 / (2.0 * math.pi * ac_hz)) if ac_hz > 0 else 0.030
    total_s = pre_s + max(first_travel_s, second_travel_s) + exposure_s + post_s

    n = int(math.ceil(total_s * fs))
    t = np.arange(n, dtype=np.float64) / fs - pre_s

    ch = []
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

        # Normalize the nominal full-scale response to leave headroom.
        recorded *= 0.82

        if noise_rms > 0:
            recorded += rng.normal(0.0, noise_rms, len(recorded))

        ch.append(recorded)

    stereo = np.column_stack(ch).astype(np.float32)

    peak = np.max(np.abs(stereo))
    if peak > 0.98:
        stereo *= 0.98 / peak

    center_spacing_mm = gate_mm - sensor_width_mm
    ideal_first_sensor_delay_s = center_spacing_mm / first_velocity
    ideal_second_sensor_delay_s = center_spacing_mm / second_velocity

    meta = {
        "exposure_s": exposure_s,
        "first_velocity_mm_s": first_velocity,
        "second_velocity_mm_s": second_velocity,
        "center_spacing_mm": center_spacing_mm,
        "ideal_first_sensor_delay_s": ideal_first_sensor_delay_s,
        "ideal_second_sensor_delay_s": ideal_second_sensor_delay_s,
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


def robust_noise_sigma(x: np.ndarray) -> float:
    # Estimate from first ~8 ms, which is pre-roll in our simulator.
    n = min(len(x), max(32, int(FS * 0.008)))
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
    """
    For exposures long enough that opening and closing transients are distinct,
    find the strongest positive opening transient and strongest negative closing
    transient in a nominal-speed-derived search window.

    Returns:
        measured exposure, opening index, closing index, confidence
    """
    x = smooth(channel.astype(np.float64), 5)
    sigma = robust_noise_sigma(x)

    # Ignore initial pre-roll.
    start = int(0.004 * fs)

    # Search opening over the early part of the trace.
    # A sensor can be delayed by curtain travel, so allow ~15 ms.
    open_end = min(len(x), start + int(0.020 * fs))
    if open_end <= start + 4:
        raise ValueError("Signal buffer too short.")

    open_idx = start + int(np.argmax(x[start:open_end]))
    open_amp = x[open_idx]

    # Search around nominal duration, allowing ±2 stops plus geometry margin.
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
    """
    Template for one finite-width sensor. Absolute time position does not matter;
    normalized cross-correlation slides it over the measured signal.
    """
    velocity = GATE_MM / (travel_ms / 1000.0)
    pre_s = 0.0015
    post_s = max(0.003, min(0.012, 2.0 / (2.0 * math.pi * ac_hz))) if ac_hz > 0 else 0.004
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
    """
    Simple normalized sliding correlation. Buffers here are intentionally small,
    so a straightforward implementation is adequate for the prototype.
    """
    x = signal.astype(np.float64)
    t = template.astype(np.float64)

    if len(t) >= len(x):
        return -1.0, 0

    t = t - np.mean(t)
    tnorm = np.linalg.norm(t)
    if tnorm == 0:
        return -1.0, 0
    t /= tnorm

    # Raw correlation then normalize local windows using cumulative sums.
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
    """
    Search candidate exposure times over ±2 stops around the selected speed.
    Intended for fast shutters where the opening/closing transients overlap.
    """
    # Restrict to useful trace region to keep template search quick.
    start = int(0.003 * fs)
    end = min(len(channel), start + int(0.030 * fs))
    segment = channel[start:end].astype(np.float64)

    # Log-spaced candidate exposures. 121 values gives useful sub-step precision.
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

    # Correlation is not perfectly bounded here because of our local normalization,
    # but clamp it to an intuitive 0..1 display value.
    confidence = float(np.clip(best_score, 0.0, 1.0))
    return best_s, best_idx, confidence


def find_peak_near(
    x: np.ndarray,
    approx_idx: int,
    radius: int,
    positive: bool,
) -> int:
    lo = max(0, approx_idx - radius)
    hi = min(len(x), approx_idx + radius + 1)
    if hi <= lo:
        return int(np.clip(approx_idx, 0, len(x) - 1))
    section = x[lo:hi]
    rel = int(np.argmax(section) if positive else np.argmin(section))
    return lo + rel


def analyze_signal(
    stereo: np.ndarray,
    nominal_s: float,
    fs: int = FS,
    expected_travel_ms: float = CURTAIN_TRAVEL_MS,
) -> AnalysisResult:
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        raise ValueError("Expected stereo samples with shape (N, 2).")

    # Time for the moving slit to sweep over the effective 2 mm sensor.
    if expected_travel_ms <= 0:
        raise ValueError("Expected curtain travel time must be positive.")

    velocity = GATE_MM / (expected_travel_ms / 1000.0)
    sensor_crossing_s = SENSOR_WIDTH_MM / velocity

    # If the exposure is comfortably longer than the sensor crossing time,
    # opening and closing transients are separable. Otherwise template fitting
    # is more robust.
    ratio = nominal_s / sensor_crossing_s
    use_transients = ratio >= 1.8

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
        # Exposure width from template matching. Separately locate the main
        # positive/negative features for approximate curtain-arrival timing.
        template_starts = []
        for c in range(2):
            exp_s, start_i, conf = template_measurement(
                stereo[:, c], fs, nominal_s, expected_travel_ms
            )
            measurements.append(exp_s)
            template_starts.append(start_i)
            confidences.append(conf)

        # For the fast regime, the template's time shift is a better first-curtain
        # marker than trying to threshold the small pulse directly.
        event_indices["sensor1_open"] = template_starts[0]
        event_indices["sensor2_open"] = template_starts[1]

        # Approximate closing time from fitted exposure. This is mainly for
        # diagnostic display; exposure comes directly from template fitting.
        event_indices["sensor1_close"] = template_starts[0] + int(measurements[0] * fs)
        event_indices["sensor2_close"] = template_starts[1] + int(measurements[1] * fs)

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
# GUI
# ---------------------------------------------------------------------------

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

# Editable presets. Service manuals usually specify full-gate curtain travel
# as a time in milliseconds; the GUI also shows the equivalent linear speed.
TRAVEL_TIME_PRESETS_MS = (2.5, 3.0, 3.3, 3.5, 4.0, 5.0, 8.0, 10.0, 12.0)


def stops_from_nominal(measured_s: float, nominal_s: float) -> float:
    """
    Signed exposure error in stops.

    Positive = measured exposure is longer than nominal (shutter is slow).
    Negative = measured exposure is shorter than nominal (shutter is fast).
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
    """Signed curtain-speed error: positive is faster, negative is slower."""
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


class ShutterTesterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Two-Point Shutter Tester")
        root.geometry("1280x820")
        root.minsize(1000, 720)

        self.samples: np.ndarray | None = None
        self.meta: dict | None = None
        self.result: AnalysisResult | None = None

        self.nominal_var = tk.StringVar(value="1/1000")
        self.actual_var = tk.StringVar(value="1/1000")
        self.follow_nominal_var = tk.BooleanVar(value=True)
        self.tolerance_var = tk.StringVar(value="±1/3 stop")
        self.expected_travel_var = tk.StringVar(value=f"{CURTAIN_TRAVEL_MS:g}")
        self.travel_tolerance_var = tk.StringVar(value="±10%")
        self.sim_first_travel_var = tk.StringVar(value=f"{CURTAIN_TRAVEL_MS:g}")
        self.sim_second_travel_var = tk.StringVar(value=f"{CURTAIN_TRAVEL_MS:g}")
        self.follow_expected_travel_var = tk.BooleanVar(value=True)

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
        self.tolerance_detail_var = tk.StringVar(value="Select a speed and run a test.")
        self.expected_speed_var = tk.StringVar(value="—")
        self.travel1_status_var = tk.StringVar(value="NOT TESTED")
        self.travel2_status_var = tk.StringVar(value="NOT TESTED")
        self.travel_detail_var = tk.StringVar(value="Set the expected full-gate travel time.")

        self._build_ui()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        outer.columnconfigure(0, minsize=340)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(outer, padding=(0, 0, 12, 0), width=340)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        main = ttk.Frame(outer, padding=(12, 0, 0, 0))
        main.grid(row=0, column=1, sticky="nsew")

        ttk.Separator(outer, orient="vertical").grid(
            row=0, column=0, sticky="nse", padx=(0, 0)
        )

        # ------------------------------------------------------------------
        # Left sidebar: shutter-speed selector
        # ------------------------------------------------------------------
        speed_frame = ttk.LabelFrame(sidebar, text="Test shutter speed", padding=9)
        speed_frame.pack(fill="x")

        ttk.Label(
            speed_frame,
            text="Select the marked shutter speed:",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 7))

        for i, speed in enumerate(COMMON_SPEEDS):
            row = 1 + i // 3
            col = i % 3

            button = tk.Radiobutton(
                speed_frame,
                text=speed,
                variable=self.nominal_var,
                value=speed,
                indicatoron=False,
                width=8,
                padx=2,
                pady=5,
                relief="raised",
                offrelief="raised",
                overrelief="ridge",
                command=self._nominal_speed_changed,
            )
            button.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
            speed_frame.columnconfigure(col, weight=1)

        exposure_settings = ttk.LabelFrame(
            sidebar, text="Exposure tolerance", padding=9
        )
        exposure_settings.pack(fill="x", pady=(9, 0))

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
        tolerance.grid(row=0, column=1, sticky="ew", pady=3)
        tolerance.bind(
            "<<ComboboxSelected>>", lambda _event: self._refresh_tolerance()
        )
        exposure_settings.columnconfigure(1, weight=1)

        # ------------------------------------------------------------------
        # Left sidebar: expected curtain travel
        # ------------------------------------------------------------------
        travel_settings = ttk.LabelFrame(
            sidebar, text="Expected curtain travel", padding=9
        )
        travel_settings.pack(fill="x", pady=(9, 0))

        ttk.Label(travel_settings, text="Full-gate time:").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=3
        )
        expected_travel = ttk.Combobox(
            travel_settings,
            textvariable=self.expected_travel_var,
            values=[f"{value:g}" for value in TRAVEL_TIME_PRESETS_MS],
            width=8,
        )
        expected_travel.grid(row=0, column=1, sticky="ew", pady=3)
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
        ).grid(row=1, column=1, columnspan=2, sticky="w", pady=3)

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
            wraplength=285,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(5, 0))
        travel_settings.columnconfigure(1, weight=1)

        self._expected_travel_changed()

        # ------------------------------------------------------------------
        # Left sidebar: simulator controls
        # ------------------------------------------------------------------
        simulator = ttk.LabelFrame(sidebar, text="Simulator input", padding=9)
        simulator.pack(fill="x", pady=(9, 0))

        ttk.Checkbutton(
            simulator,
            text="Follow selected shutter speed",
            variable=self.follow_nominal_var,
            command=self._follow_nominal_changed,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Label(simulator, text="Actual exposure:").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self.actual_combo = ttk.Combobox(
            simulator,
            textvariable=self.actual_var,
            values=COMMON_SPEEDS,
            width=11,
        )
        self.actual_combo.grid(row=1, column=1, sticky="ew", pady=3)

        ttk.Checkbutton(
            simulator,
            text="Curtains follow expected travel",
            variable=self.follow_expected_travel_var,
            command=self._follow_expected_travel_changed,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 4))

        ttk.Label(simulator, text="1st curtain:").grid(
            row=3, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self.sim_first_travel_entry = ttk.Entry(
            simulator,
            textvariable=self.sim_first_travel_var,
            width=9,
        )
        self.sim_first_travel_entry.grid(row=3, column=1, sticky="ew", pady=3)

        ttk.Label(simulator, text="2nd curtain:").grid(
            row=4, column=0, sticky="w", padx=(0, 6), pady=3
        )
        self.sim_second_travel_entry = ttk.Entry(
            simulator,
            textvariable=self.sim_second_travel_var,
            width=9,
        )
        self.sim_second_travel_entry.grid(row=4, column=1, sticky="ew", pady=3)

        ttk.Label(
            simulator,
            text="Curtain values are full-gate times in ms.",
            wraplength=285,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))

        simulator.columnconfigure(1, weight=1)

        test_button = ttk.Button(
            sidebar,
            text="TEST SIMULATED SHUTTER",
            command=self.run_test,
        )
        test_button.pack(fill="x", pady=(12, 0), ipady=7)

        ttk.Button(
            sidebar,
            text="Repeat with new noise",
            command=self.run_test,
        ).pack(fill="x", pady=(6, 0), ipady=3)

        self._follow_nominal_changed()
        self._follow_expected_travel_changed()

        # ------------------------------------------------------------------
        # Right side: large overall result
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Separate first- and second-curtain travel verdicts
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Detailed measurement values
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------------
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

        # Show an initial result immediately.
        self.root.after(100, self.run_test)

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

    def run_test(self):
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
                    self.sim_first_travel_var.get(), "Simulated 1st-curtain travel"
                )
                actual_second_travel_ms = self._parse_positive_ms(
                    self.sim_second_travel_var.get(), "Simulated 2nd-curtain travel"
                )

            self.status_var.set("Generating simulated stereo signal…")
            self.root.update_idletasks()

            samples, meta = simulate_shutter(
                actual_s,
                travel_ms=actual_first_travel_ms,
                second_travel_ms=actual_second_travel_ms,
            )

            self.status_var.set("Analyzing…")
            self.root.update_idletasks()

            result = analyze_signal(
                samples,
                nominal_s,
                expected_travel_ms=expected_travel_ms,
            )

            self.samples = samples
            self.meta = meta
            self.result = result

            self._update_results(result, nominal_s)
            self._update_plot(samples, result)

            actual_text = format_speed(actual_s)
            measured_text = format_speed(result.exposure_average_s)
            error_stops = stops_from_nominal(result.exposure_average_s, nominal_s)

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
            r.first_curtain_full_travel_s, expected_travel_s
        )
        travel2_error = travel_error_percent(
            r.second_curtain_full_travel_s, expected_travel_s
        )

        self.travel1_var.set(
            format_travel_measurement(
                r.first_curtain_full_travel_s, expected_travel_s
            )
        )
        self.travel2_var.set(
            format_travel_measurement(
                r.second_curtain_full_travel_s, expected_travel_s
            )
        )

        self.mismatch_var.set(f"{r.sensor_mismatch_percent:.2f}%")
        self.conf_var.set(f"{r.confidence * 100:.0f}%")

        tolerance = self._get_tolerance_stops()
        travel_tolerance = self._get_travel_tolerance_percent()

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
            f"({curtain_speed_m_s(expected_travel_s):.2f} m/s), speed tolerance "
            f"{self.travel_tolerance_var.get()}. Positive speed error means a "
            "faster curtain; negative means a slower curtain."
        )

        # Require both sides of the frame to be within tolerance, not only
        # their average. This catches a shutter whose average is correct but
        # which has significant side-to-side exposure error.
        exposure_passed = (
            np.isfinite(s1_stops)
            and np.isfinite(s2_stops)
            and abs(s1_stops) <= tolerance
            and abs(s2_stops) <= tolerance
        )

        # The overall verdict covers both exposure accuracy and the independent
        # travel-speed checks. Separate curtain indicators above show which part
        # caused a failure.
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

    def _update_plot(self, samples: np.ndarray, r: AnalysisResult):
        self.ax.clear()

        t_ms = np.arange(len(samples)) / FS * 1000.0
        self.ax.plot(t_ms, samples[:, 0], label="Sensor 1", linewidth=1.0)
        self.ax.plot(t_ms, samples[:, 1], label="Sensor 2", linewidth=1.0)

        markers = [
            ("sensor1_open", "S1 open"),
            ("sensor1_close", "S1 close"),
            ("sensor2_open", "S2 open"),
            ("sensor2_close", "S2 close"),
        ]

        for key, label in markers:
            idx = r.event_indices.get(key)
            if idx is not None and 0 <= idx < len(samples):
                self.ax.axvline(
                    idx / FS * 1000.0,
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.55,
                )

        # Zoom to the active portion, but leave useful context.
        active = list(r.event_indices.values())
        if active:
            lo = max(0, min(active) - int(0.004 * FS))
            hi = min(len(samples) - 1, max(active) + int(0.008 * FS))
            if hi > lo:
                self.ax.set_xlim(t_ms[lo], t_ms[hi])

        self.ax.set_title(
            f"96 kHz simulated AC-coupled input — analyzer: {r.method}"
        )
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Line-in amplitude")
        self.ax.grid(True, alpha=0.25)
        self.ax.legend(loc="upper right")
        self.figure.tight_layout()
        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    ShutterTesterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
