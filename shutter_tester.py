#!/usr/bin/env python3
"""
Two-sensor focal-plane shutter tester, v12.

Key features
------------
* Simulator and continuous live stereo audio input.
* Live sources start listening automatically; there are no ARM/CANCEL controls.
* Simulator controls are shown only when Simulator is selected.
* AC-coupled (default) and DC/direct signal modes.
* Searchable, alphabetically sorted camera preset selector.
* Camera preset database loaded from camera_presets.csv beside the script.
* Service-manual travel ranges and curtain mismatch limits can be stored directly in the CSV.
* Large Nikon focal-plane-film-camera preset catalogue, with FM2 and FM2N
  intentionally kept separate (1/200 vs 1/250 X-sync).
* Separate optical aperture width and measured sensor center-to-center spacing.
* Independent exposure measurement at both sensors plus 1st/2nd curtain travel.
* Automatic camera-specific curtain-travel tolerance where service data is known; ±5% fallback.
* Separate curtain-to-curtain speed-mismatch assessment, with service-manual limits when available.
* Selected shutter speed is a search hint, not a forced result.
* ±3-stop analysis window for initial tensioning / badly misadjusted shutters.

Dependencies
------------
    pip install numpy matplotlib sounddevice

sounddevice is only required for live audio; Simulator mode works without it.
"""
from __future__ import annotations

import csv
import math
import time
import threading
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk, messagebox
from typing import Optional

import numpy as np

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError as exc:
    raise SystemExit("Missing matplotlib. Install with: pip install numpy matplotlib") from exc

try:
    import sounddevice as sd
except Exception:
    sd = None


# ---------------------------------------------------------------------------
# Core defaults
# ---------------------------------------------------------------------------

FS = 96_000
DEFAULT_GATE_MM = 24.0
DEFAULT_APERTURE_MM = 2.0
DEFAULT_SPACING_MM = 22.0
DEFAULT_TRAVEL_MS = 3.3
DEFAULT_AC_HZ = 35.0
SENSOR_TAU_US = 7.0
NOISE_RMS = 0.003
SEARCH_STOPS = 3.0

COMMON_SPEEDS = [
    "1", "1/2", "1/4", "1/8", "1/15", "1/30", "1/60", "1/80",
    "1/90", "1/100", "1/125", "1/200", "1/250", "1/500", "1/1000",
    "1/2000", "1/4000", "1/8000",
]

TOLERANCE_OPTIONS = {
    "±1/6 stop": 1.0 / 6.0,
    "±1/3 stop": 1.0 / 3.0,
    "±1/2 stop": 1.0 / 2.0,
    "±1 stop": 1.0,
}

TRAVEL_TOLERANCE_OPTIONS = {
    "Auto (service / ±5%)": None,
    "±2%": 2.0,
    "±5%": 5.0,
    "±10%": 10.0,
    "±15%": 15.0,
    "±20%": 20.0,
}

MISMATCH_TOLERANCE_OPTIONS = {
    "Auto (service / ±5%)": None,
    "±2%": 2.0,
    "±5%": 5.0,
    "±10%": 10.0,
    "±15%": 15.0,
}
GENERIC_TRAVEL_TOLERANCE_PERCENT = 5.0

# Historical/manual presets kept in addition to camera-specific values.
TRAVEL_PRESETS = [
    ("2.0 ms — very fast late metal/carbon shutter", 2.0),
    ("2.5 ms — fast pro metal shutter", 2.5),
    ("2.9 ms — Nikon F4 service value", 2.9),
    ("3.3 ms — FE2 / high-speed Nikon generation", 3.3),
    ("3.6 ms — original Nikon FM2", 3.6),
    ("4.0 ms — fast metal blade", 4.0),
    ("5.0 ms — medium-fast metal blade", 5.0),
    ("7.0 ms — older vertical metal blade", 7.0),
    ("8.0 ms — older vertical shutter", 8.0),
    ("10.0 ms — Nikon F2/F3 class horizontal foil", 10.0),
    ("12.0 ms — medium horizontal curtain", 12.0),
    ("14.5 ms — Nikon F / SP-generation horizontal", 14.5),
    ("18.0 ms — early Nikon horizontal cloth", 18.0),
]


# ---------------------------------------------------------------------------
# Camera presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraPreset:
    name: str
    path_mm: float  # reference distance used for the stored travel time
    x_sync: str
    travel_ms: Optional[float]
    travel_quality: str  # "factory", "family", "estimate", "unknown"
    note: str = ""
    service_travel_tolerance_pct: Optional[float] = None
    service_travel_min_ms: Optional[float] = None
    service_travel_max_ms: Optional[float] = None
    service_mismatch_max_ms: Optional[float] = None
    source_url: str = ""


CAMERA_PRESET_FILENAME = "camera_presets.csv"
_ALLOWED_TRAVEL_QUALITIES = {"factory", "family", "estimate", "unknown"}


def _optional_float(value: str, field: str, line_no: int) -> Optional[float]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(
            f"{CAMERA_PRESET_FILENAME}:{line_no}: {field} must be a number or blank; got {value!r}."
        ) from exc
    if result <= 0:
        raise ValueError(
            f"{CAMERA_PRESET_FILENAME}:{line_no}: {field} must be greater than zero; got {result:g}."
        )
    return result


def _camera_preset_path() -> Path:
    # Normally the CSV lives beside the Python file. The CWD fallback is handy
    # when running the script from an IDE or an unpacked bundle.
    candidates = [
        Path(__file__).resolve().with_name(CAMERA_PRESET_FILENAME),
        Path.cwd() / CAMERA_PRESET_FILENAME,
    ]
    for path in candidates:
        if path.is_file():
            return path
    searched = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find {CAMERA_PRESET_FILENAME}. Keep it beside this Python file.\n"
        f"Searched:\n  - {searched}"
    )


def load_camera_presets() -> list[CameraPreset]:
    path = _camera_preset_path()
    required = {
        "name",
        "x_sync",
        "travel_ms",
        "travel_quality",
        "service_travel_tolerance_pct",
        "note",
    }
    presets: list[CameraPreset] = []
    seen_names: set[str] = set()

    # utf-8-sig accepts both normal UTF-8 and Excel-style UTF-8-with-BOM files.
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"{path.name} is missing required column(s): {', '.join(sorted(missing))}."
            )
        # v12 calls this what it really is.  Keep v11 path_mm compatibility so
        # an older hand-maintained CSV still opens.
        if "reference_distance_mm" in fieldnames:
            distance_col = "reference_distance_mm"
        elif "path_mm" in fieldnames:
            distance_col = "path_mm"
        else:
            raise ValueError(
                f"{path.name} needs reference_distance_mm (preferred) or legacy path_mm."
            )

        for line_no, row in enumerate(reader, start=2):
            if not any((v or "").strip() for v in row.values()):
                continue

            name = (row.get("name") or "").strip()
            if not name:
                raise ValueError(f"{path.name}:{line_no}: name may not be blank.")
            key = name.casefold()
            if key in seen_names:
                raise ValueError(f"{path.name}:{line_no}: duplicate camera name {name!r}.")
            seen_names.add(key)

            distance_text = (row.get(distance_col) or "").strip()
            try:
                path_mm = float(distance_text)
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}:{line_no}: {distance_col} must be a number; got {distance_text!r}."
                ) from exc
            if path_mm <= 0:
                raise ValueError(f"{path.name}:{line_no}: {distance_col} must be greater than zero.")

            quality = (row.get("travel_quality") or "unknown").strip().lower() or "unknown"
            if quality not in _ALLOWED_TRAVEL_QUALITIES:
                allowed = ", ".join(sorted(_ALLOWED_TRAVEL_QUALITIES))
                raise ValueError(
                    f"{path.name}:{line_no}: travel_quality must be one of {allowed}; got {quality!r}."
                )

            service_min = _optional_float(
                row.get("service_travel_min_ms") or "", "service_travel_min_ms", line_no
            )
            service_max = _optional_float(
                row.get("service_travel_max_ms") or "", "service_travel_max_ms", line_no
            )
            if service_min is not None and service_max is not None and service_min > service_max:
                raise ValueError(
                    f"{path.name}:{line_no}: service_travel_min_ms cannot exceed service_travel_max_ms."
                )

            presets.append(
                CameraPreset(
                    name=name,
                    path_mm=path_mm,
                    x_sync=(row.get("x_sync") or "").strip(),
                    travel_ms=_optional_float(row.get("travel_ms") or "", "travel_ms", line_no),
                    travel_quality=quality,
                    note=(row.get("note") or "").strip(),
                    service_travel_tolerance_pct=_optional_float(
                        row.get("service_travel_tolerance_pct") or "",
                        "service_travel_tolerance_pct",
                        line_no,
                    ),
                    service_travel_min_ms=service_min,
                    service_travel_max_ms=service_max,
                    service_mismatch_max_ms=_optional_float(
                        row.get("service_mismatch_max_ms") or "",
                        "service_mismatch_max_ms",
                        line_no,
                    ),
                    source_url=(row.get("source_url") or "").strip(),
                )
            )

    if not presets:
        raise ValueError(f"{path.name} contains no camera presets.")
    return sorted(presets, key=lambda p: p.name.casefold())


CAMERA_PRESETS = load_camera_presets()
CAMERA_BY_NAME = {p.name: p for p in CAMERA_PRESETS}
CAMERA_NAMES = [p.name for p in CAMERA_PRESETS]


# ---------------------------------------------------------------------------
# Utility / signal processing
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
    d = 1.0 / seconds
    return f"1/{d:.2f} s" if d < 10 else f"1/{d:.0f} s"


def one_pole_lowpass(x, fs, tau_s):
    if tau_s <= 0:
        return np.asarray(x, dtype=float).copy()
    a = 1.0 - math.exp(-1.0 / (fs * tau_s))
    y = np.empty_like(x, dtype=float)
    y[0] = a * x[0]
    for i in range(1, len(x)):
        y[i] = y[i-1] + a * (x[i] - y[i-1])
    return y


def one_pole_highpass(x, fs, cutoff_hz):
    if cutoff_hz <= 0:
        return np.asarray(x, dtype=float).copy()
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / fs
    a = rc / (rc + dt)
    y = np.zeros_like(x, dtype=float)
    for i in range(1, len(x)):
        y[i] = a * (y[i-1] + x[i] - x[i-1])
    return y


def smooth(x, width=5):
    if width <= 1:
        return np.asarray(x, dtype=float).copy()
    k = np.ones(width) / width
    return np.convolve(np.asarray(x, dtype=float), k, mode="same")


def robust_sigma(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 8:
        return max(float(np.std(x)), 1e-7)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return max(1e-7, 1.4826 * mad)


def stops_from_nominal(measured, nominal):
    if measured <= 0 or nominal <= 0 or not np.isfinite(measured):
        return float("nan")
    return math.log2(measured / nominal)


def format_stop_error(stops):
    if not np.isfinite(stops):
        return "—"
    direction = "slow / more exposure" if stops > 0 else "fast / less exposure" if stops < 0 else "on target"
    return f"{stops:+.2f} stops ({direction})"


def curtain_speed_m_s(travel_s, path_mm):
    if travel_s <= 0:
        return float("nan")
    return (path_mm / 1000.0) / travel_s


def travel_error_percent(measured_s, expected_s):
    if measured_s <= 0 or expected_s <= 0:
        return float("nan")
    # same path cancels: speed ratio = expected_time / measured_time
    return 100.0 * (expected_s / measured_s - 1.0)


def curtain_speed_mismatch_percent(first_travel_s, second_travel_s):
    """Symmetric percentage difference between curtain speeds.

    For the same travel path, speed is inverse to travel time.  The symmetric
    speed difference simplifies to 2*|t1-t2|/(t1+t2).
    """
    if first_travel_s <= 0 or second_travel_s <= 0:
        return float("nan")
    return 200.0 * abs(first_travel_s - second_travel_s) / (first_travel_s + second_travel_s)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def finite_sensor_light(t, x_center_mm, aperture_mm, first_v, second_v, exposure_s, samples=120):
    half = aperture_mm / 2.0
    xs = np.linspace(x_center_mm - half, x_center_mm + half, samples)
    open_t = xs / first_v
    close_t = exposure_s + xs / second_v
    lit = (t[:, None] >= open_t[None, :]) & (t[:, None] < close_t[None, :])
    return lit.mean(axis=1)


def simulate_shutter(exposure_s, travel1_ms, travel2_ms, path_mm, spacing_mm, aperture_mm,
                     coupling="AC coupled", fs=FS, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    v1 = path_mm / (travel1_ms / 1000.0)
    v2 = path_mm / (travel2_ms / 1000.0)

    # Put the two sensor centers symmetrically around the frame center.
    x1 = max(aperture_mm / 2.0, (path_mm - spacing_mm) / 2.0)
    x2 = min(path_mm - aperture_mm / 2.0, x1 + spacing_mm)

    pre = 0.020
    post = 0.050
    total = pre + exposure_s + max(travel1_ms, travel2_ms) / 1000.0 + post
    n = max(256, int(math.ceil(total * fs)))
    t = np.arange(n) / fs - pre

    chs = []
    for xc in (x1, x2):
        light = finite_sensor_light(t, xc, aperture_mm, v1, v2, exposure_s)
        y = one_pole_lowpass(light, fs, SENSOR_TAU_US * 1e-6)
        if coupling.startswith("AC"):
            y = one_pole_highpass(y, fs, DEFAULT_AC_HZ)
        y *= 0.82
        y += rng.normal(0.0, NOISE_RMS, len(y))
        chs.append(y)

    stereo = np.column_stack(chs).astype(np.float32)
    peak = np.max(np.abs(stereo))
    if peak > 0.98:
        stereo *= 0.98 / peak
    return stereo


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    method: str
    exposure_sensor_1_s: float
    exposure_sensor_2_s: float
    exposure_average_s: float
    first_sensor_delay_s: float
    second_sensor_delay_s: float
    first_curtain_full_travel_s: float
    second_curtain_full_travel_s: float
    sensor_mismatch_percent: float
    confidence: float
    event_indices: dict


def _edge_measure_ac(channel, fs, nominal_s):
    x = smooth(channel, 5)
    n0 = min(len(x), max(64, int(0.008 * fs)))
    sigma = robust_sigma(x[:n0])
    start = min(len(x)-1, int(0.002 * fs))
    open_end = min(len(x), start + int(0.040 * fs))
    if open_end <= start + 4:
        raise ValueError("Signal window too short")

    # Allow either polarity because external electronics may invert the signal.
    section = x[start:open_end]
    pos_i = int(np.argmax(section)) + start
    neg_i = int(np.argmin(section)) + start
    open_idx = pos_i if abs(x[pos_i]) >= abs(x[neg_i]) else neg_i
    opening_sign = 1.0 if x[open_idx] >= 0 else -1.0

    low = max(4 / fs, nominal_s / (2.0 ** SEARCH_STOPS))
    high = nominal_s * (2.0 ** SEARCH_STOPS)
    close_start = max(open_idx + 2, open_idx + int(low * fs))
    close_end = min(len(x), open_idx + int(high * fs) + int(0.060 * fs))
    if close_end <= close_start + 4:
        raise ValueError("Closing edge not yet in live buffer")

    close_section = x[close_start:close_end]
    # AC opening and closing transients are normally opposite polarity.
    if opening_sign > 0:
        close_idx = int(np.argmin(close_section)) + close_start
    else:
        close_idx = int(np.argmax(close_section)) + close_start

    measured = (close_idx - open_idx) / fs
    snr = min(abs(x[open_idx]), abs(x[close_idx])) / sigma
    conf = float(np.clip((snr - 3.0) / 15.0, 0.0, 1.0))
    return measured, open_idx, close_idx, conf



def _make_ac_template(exposure_s, fs, expected_travel_ms, path_mm, aperture_mm):
    """Build one finite-aperture AC-coupled sensor pulse for fast-shutter fitting."""
    expected_s = expected_travel_ms / 1000.0
    if expected_s <= 0 or path_mm <= 0 or aperture_mm <= 0:
        raise ValueError("Expected travel, path and optical aperture must be positive")
    velocity = path_mm / expected_s
    pre_s = 0.0015
    post_s = 0.010
    crossing_s = aperture_mm / velocity
    total_s = pre_s + crossing_s + exposure_s + post_s
    n = max(64, int(math.ceil(total_s * fs)))
    t = np.arange(n, dtype=float) / fs - pre_s
    light = finite_sensor_light(
        t,
        aperture_mm / 2.0,
        aperture_mm,
        velocity,
        velocity,
        exposure_s,
        samples=100,
    )
    y = one_pole_lowpass(light, fs, SENSOR_TAU_US * 1e-6)
    y = one_pole_highpass(y, fs, DEFAULT_AC_HZ)
    y -= np.mean(y)
    return y, int(round(pre_s * fs))


def _best_normalized_match(signal, template):
    """Normalized sliding correlation; returns score and start index."""
    x = np.asarray(signal, dtype=float)
    t = np.asarray(template, dtype=float)
    if len(t) >= len(x) or len(t) < 4:
        return -1.0, 0
    t = t - np.mean(t)
    tnorm = float(np.linalg.norm(t))
    if tnorm <= 1e-12:
        return -1.0, 0
    t /= tnorm
    corr = np.correlate(x, t, mode="valid")
    n = len(t)
    cs = np.concatenate(([0.0], np.cumsum(x)))
    cs2 = np.concatenate(([0.0], np.cumsum(x*x)))
    sums = cs[n:] - cs[:-n]
    sums2 = cs2[n:] - cs2[:-n]
    energy = sums2 - (sums*sums)/n
    denom = np.sqrt(np.maximum(energy, 1e-12))
    score = corr / denom
    best = int(np.argmax(score))
    return float(score[best]), best


def _template_measure_ac(channel, fs, nominal_s, expected_travel_ms, path_mm, aperture_mm):
    """
    Fit the complete AC pulse when opening/closing transients overlap.

    The selected shutter speed only defines a ±SEARCH_STOPS candidate range;
    it is not imposed on the reported result.
    """
    x = np.asarray(channel, dtype=float)
    if len(x) < 128:
        raise ValueError("Signal window too short")

    n0 = min(len(x), max(64, int(0.008 * fs)))
    baseline = float(np.median(x[:n0]))
    centered = x - baseline
    strongest = int(np.argmax(np.abs(centered)))

    # Fast-shutter events are compact. Crop around the strongest transient so
    # live ring buffers do not make the candidate search unnecessarily costly.
    lo = max(0, strongest - int(0.012 * fs))
    max_candidate_s = nominal_s * (2.0 ** SEARCH_STOPS)
    hi = min(len(x), strongest + int((max_candidate_s + 0.025) * fs))
    segment = centered[lo:hi]
    if len(segment) < 128:
        segment = centered
        lo = 0

    candidates = nominal_s * (2.0 ** np.linspace(-SEARCH_STOPS, SEARCH_STOPS, 169))
    best_score = -1e9
    best_s = nominal_s
    best_start = 0
    best_open_offset = 0

    for exposure_s in candidates:
        template, open_offset = _make_ac_template(
            float(exposure_s), fs, expected_travel_ms, path_mm, aperture_mm
        )
        score, start = _best_normalized_match(segment, template)
        if score > best_score:
            best_score = score
            best_s = float(exposure_s)
            best_start = lo + start
            best_open_offset = open_offset

    open_idx = best_start + best_open_offset
    close_idx = open_idx + int(round(best_s * fs))
    confidence = float(np.clip(best_score, 0.0, 1.0))
    return best_s, open_idx, close_idx, confidence


def _edge_measure_dc(channel, fs, nominal_s):
    x = smooth(channel, 7)
    n0 = min(len(x), max(64, int(0.008 * fs)))
    baseline = float(np.median(x[:n0]))
    centered = x - baseline
    sigma = robust_sigma(centered[:n0])

    # Determine dominant illuminated polarity from largest excursion.
    if abs(np.max(centered)) >= abs(np.min(centered)):
        y = centered
    else:
        y = -centered

    peak = float(np.max(y))
    if peak < max(6 * sigma, 1e-4):
        raise ValueError("No clear shutter signal")
    threshold = max(0.35 * peak, 4 * sigma)
    mask = y > threshold
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        raise ValueError("Could not find both DC edges")

    # Restrict to a region around the expected duration but don't force the result.
    open_idx = int(idx[0])
    min_close = open_idx + int(max(3/fs, nominal_s / (2 ** SEARCH_STOPS)) * fs)
    max_close = min(len(y)-1, open_idx + int(nominal_s * (2 ** SEARCH_STOPS) * fs) + int(.05*fs))
    candidates = idx[(idx >= min_close) & (idx <= max_close)]
    close_idx = int(candidates[-1] if len(candidates) else idx[-1])
    measured = (close_idx - open_idx) / fs
    conf = float(np.clip((peak / sigma - 5) / 20, 0, 1))
    return measured, open_idx, close_idx, conf


def analyze_signal(stereo, nominal_s, expected_travel_ms, path_mm, spacing_mm, aperture_mm,
                   coupling="AC coupled", fs=FS):
    stereo = np.asarray(stereo)
    if stereo.ndim != 2 or stereo.shape[1] < 2:
        raise ValueError("Expected stereo input with two channels")
    stereo = stereo[:, :2]

    # A finite optical aperture smears very fast shutter pulses. When the
    # selected speed is short enough for opening/closing AC transients to
    # overlap, fit the entire expected pulse shape. Otherwise use edges.
    expected_travel_s = expected_travel_ms / 1000.0
    sensor_crossing_s = aperture_mm * expected_travel_s / path_mm
    use_template = (
        coupling.startswith("AC")
        and sensor_crossing_s > 0
        and nominal_s / sensor_crossing_s < 1.8
    )

    measurements, opens, closes, confs = [], [], [], []
    for c in range(2):
        if use_template:
            exp, oi, ci, conf = _template_measure_ac(
                stereo[:, c], fs, nominal_s, expected_travel_ms, path_mm, aperture_mm
            )
        elif coupling.startswith("AC"):
            exp, oi, ci, conf = _edge_measure_ac(stereo[:, c], fs, nominal_s)
        else:
            exp, oi, ci, conf = _edge_measure_dc(stereo[:, c], fs, nominal_s)
        measurements.append(exp)
        opens.append(oi)
        closes.append(ci)
        confs.append(conf)

    exp1, exp2 = measurements
    avg = (exp1 + exp2) / 2.0
    first_delay = (opens[1] - opens[0]) / fs
    second_delay = (closes[1] - closes[0]) / fs

    if spacing_mm <= 0 or path_mm <= 0:
        raise ValueError("Travel path and sensor spacing must be positive")

    def delay_to_full(delay_s):
        if abs(delay_s) < 1.0 / fs:
            return float("nan")
        v_mm_s = spacing_mm / abs(delay_s)
        return path_mm / v_mm_s

    first_full = delay_to_full(first_delay)
    second_full = delay_to_full(second_delay)
    mismatch = 100.0 * abs(exp1 - exp2) / avg if avg > 0 else float("nan")

    return AnalysisResult(
        method=("AC template fit" if use_template else
                "AC transient edges" if coupling.startswith("AC") else "DC level edges"),
        exposure_sensor_1_s=exp1,
        exposure_sensor_2_s=exp2,
        exposure_average_s=avg,
        first_sensor_delay_s=first_delay,
        second_sensor_delay_s=second_delay,
        first_curtain_full_travel_s=first_full,
        second_curtain_full_travel_s=second_full,
        sensor_mismatch_percent=mismatch,
        confidence=float(np.mean(confs)),
        event_indices={
            "sensor1_open": opens[0], "sensor2_open": opens[1],
            "sensor1_close": closes[0], "sensor2_close": closes[1],
        },
    )


# ---------------------------------------------------------------------------
# Searchable combobox
# ---------------------------------------------------------------------------

class SearchableCombobox(ttk.Combobox):
    def __init__(self, master=None, *, values=(), **kwargs):
        self._all_values = list(values)
        super().__init__(master, values=self._all_values, **kwargs)
        self.bind("<KeyRelease>", self._on_keyrelease, add=True)
        self.bind("<FocusIn>", self._on_focus, add=True)

    def set_all_values(self, values):
        self._all_values = list(values)
        self["values"] = self._all_values

    def _on_focus(self, _event=None):
        if not self.get().strip():
            self["values"] = self._all_values

    def _on_keyrelease(self, event):
        if event.keysym in {"Up", "Down", "Left", "Right", "Return", "Escape", "Tab"}:
            return
        query = self.get().strip().casefold()
        if not query:
            filtered = self._all_values
        else:
            tokens = query.split()
            filtered = [v for v in self._all_values if all(t in v.casefold() for t in tokens)]
        self["values"] = filtered
        if filtered:
            try:
                self.event_generate("<Down>")
            except tk.TclError:
                pass


# ---------------------------------------------------------------------------
# GUI application
# ---------------------------------------------------------------------------

class ShutterTesterApp:
    def __init__(self, root):
        self.root = root
        root.title("Two-Point Shutter Tester v12")
        root.geometry("1320x820")
        root.minsize(1050, 680)

        self.samples = None
        self.result = None
        self.audio_stream = None
        self.live_chunks = deque()
        self.live_samples = 0
        self.live_lock = threading.Lock()
        self.live_trigger_time = None
        self.live_cooldown_until = 0.0
        self.live_peak_floor = 0.025
        self.last_live_error = ""
        self.device_map = {}

        self.nominal_var = tk.StringVar(value="1/1000")
        self.tolerance_var = tk.StringVar(value="±1/3 stop")
        self.camera_var = tk.StringVar(value="Nikon FE2")
        self.camera_info_var = tk.StringVar(value="")
        self.path_var = tk.StringVar(value="24")
        self.spacing_var = tk.StringVar(value="22")
        self.aperture_var = tk.StringVar(value="2")
        self.expected_travel_var = tk.StringVar(value="3.3")
        self.travel_tolerance_var = tk.StringVar(value="Auto (service / ±5%)")
        self.mismatch_tolerance_var = tk.StringVar(value="Auto (service / ±5%)")
        self.coupling_var = tk.StringVar(value="AC coupled")

        self.source_var = tk.StringVar(value="Simulator")
        self.actual_var = tk.StringVar(value="1/1000")
        self.follow_nominal_var = tk.BooleanVar(value=True)
        self.sim_first_var = tk.StringVar(value="3.3")
        self.sim_second_var = tk.StringVar(value="3.3")
        self.follow_travel_var = tk.BooleanVar(value=True)

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
        self.travel1_status_var = tk.StringVar(value="NOT TESTED")
        self.travel2_status_var = tk.StringVar(value="NOT TESTED")
        self.curtain_mismatch_var = tk.StringVar(value="—")
        self.curtain_mismatch_status_var = tk.StringVar(value="NOT TESTED")
        self.tolerance_detail_var = tk.StringVar(value="Select a speed and fire the shutter.")

        self._build_ui()
        self._refresh_input_devices()
        self._apply_camera_preset()
        self._source_changed()
        self.root.after(80, self._poll_live)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------- UI helpers -------------------------
    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, minsize=390)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        # Scrollable sidebar that only scrolls when content is actually taller.
        side_wrap = ttk.Frame(outer, width=390)
        side_wrap.grid(row=0, column=0, sticky="nsew")
        side_wrap.grid_propagate(False)
        side_canvas = tk.Canvas(side_wrap, highlightthickness=0, borderwidth=0)
        side_scroll = ttk.Scrollbar(side_wrap, orient="vertical", command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_scroll.set)
        side_canvas.pack(side="left", fill="both", expand=True)
        self.side_scroll = side_scroll
        self.side_canvas = side_canvas

        sidebar = ttk.Frame(side_canvas, padding=(0, 0, 8, 0))
        self.side_window = side_canvas.create_window((0, 0), window=sidebar, anchor="nw")

        def sync_scroll(_evt=None):
            side_canvas.itemconfigure(self.side_window, width=side_canvas.winfo_width())
            side_canvas.configure(scrollregion=side_canvas.bbox("all"))
            bbox = side_canvas.bbox("all")
            need = bool(bbox and (bbox[3] - bbox[1]) > side_canvas.winfo_height() + 2)
            if need:
                if not side_scroll.winfo_ismapped():
                    side_scroll.pack(side="right", fill="y")
            else:
                if side_scroll.winfo_ismapped():
                    side_scroll.pack_forget()
                side_canvas.yview_moveto(0)
        sidebar.bind("<Configure>", sync_scroll)
        side_canvas.bind("<Configure>", sync_scroll)
        def camera_popdown_open():
            # ttk's drop-down is a separate toplevel.  Because the sidebar's
            # wheel handler is registered globally, explicitly leave the
            # wheel alone while this popup is mapped; the popup list then
            # receives the wheel without the settings canvas moving behind it.
            combo = getattr(self, "camera_combo", None)
            if combo is None:
                return False
            try:
                popdown = self.root.tk.call("ttk::combobox::PopdownWindow", str(combo))
                return bool(int(self.root.tk.call("winfo", "ismapped", popdown)))
            except (tk.TclError, ValueError):
                return False

        def pointer_is_in_sidebar(event):
            try:
                widget = self.root.winfo_containing(event.x_root, event.y_root)
            except (tk.TclError, AttributeError):
                return False
            while widget is not None:
                if widget in (sidebar, side_canvas, side_wrap):
                    return True
                widget = getattr(widget, "master", None)
            return False

        def wheel(event):
            if camera_popdown_open() or not pointer_is_in_sidebar(event):
                return
            bbox = side_canvas.bbox("all")
            if not bbox or (bbox[3] - bbox[1]) <= side_canvas.winfo_height() + 2:
                return
            delta = getattr(event, "delta", 0)
            if delta:
                steps = int(-delta / 120)
                if steps == 0:
                    steps = -1 if delta > 0 else 1
            else:
                steps = -1 if getattr(event, "num", None) == 4 else 1
            side_canvas.yview_scroll(steps, "units")

        side_canvas.bind_all("<MouseWheel>", wheel, add="+")
        side_canvas.bind_all("<Button-4>", wheel, add="+")
        side_canvas.bind_all("<Button-5>", wheel, add="+")

        main = ttk.Frame(outer, padding=(10, 0, 0, 0))
        main.grid(row=0, column=1, sticky="nsew")

        # Input source
        src = ttk.LabelFrame(sidebar, text="Input", padding=7)
        src.pack(fill="x")
        ttk.Label(src, text="Source:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        self.source_combo = ttk.Combobox(src, textvariable=self.source_var, state="readonly")
        self.source_combo.grid(row=0, column=1, sticky="ew", pady=2)
        self.source_combo.bind("<<ComboboxSelected>>", lambda _e: self._source_changed())
        ttk.Button(src, text="↻", width=3, command=self._refresh_input_devices).grid(row=0, column=2, padx=(4, 0))
        ttk.Label(src, text="Signal:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
        coupling = ttk.Combobox(src, textvariable=self.coupling_var,
                                values=["AC coupled", "DC / direct coupled"], state="readonly")
        coupling.grid(row=1, column=1, columnspan=2, sticky="ew", pady=2)
        coupling.bind("<<ComboboxSelected>>", lambda _e: self._clear_live_state())
        ttk.Label(src, text="Live sources listen continuously; fire the camera whenever ready.",
                  wraplength=330).grid(row=2, column=0, columnspan=3, sticky="w", pady=(3,0))
        src.columnconfigure(1, weight=1)

        # Camera preset
        cam = ttk.LabelFrame(sidebar, text="Camera / shutter preset", padding=7)
        cam.pack(fill="x", pady=(7, 0))
        ttk.Label(cam, text="Camera:").grid(row=0, column=0, sticky="w", padx=(0,6))
        self.camera_combo = SearchableCombobox(cam, textvariable=self.camera_var, values=CAMERA_NAMES)
        self.camera_combo.grid(row=0, column=1, sticky="ew")
        self.camera_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_camera_preset())
        self.camera_combo.bind("<Return>", lambda _e: self._apply_camera_preset())
        ttk.Label(cam, textvariable=self.camera_info_var, wraplength=330).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        cam.columnconfigure(1, weight=1)

        # Shutter speeds
        speeds = ttk.LabelFrame(sidebar, text="Test shutter speed", padding=7)
        speeds.pack(fill="x", pady=(7, 0))
        for i, speed in enumerate(COMMON_SPEEDS):
            b = tk.Radiobutton(speeds, text=speed, variable=self.nominal_var, value=speed,
                               indicatoron=False, width=7, padx=1, pady=3, relief="raised",
                               command=self._nominal_changed)
            r, c = divmod(i, 4)
            b.grid(row=r, column=c, sticky="ew", padx=1, pady=1)
            speeds.columnconfigure(c, weight=1)

        # Geometry and travel
        geom = ttk.LabelFrame(sidebar, text="Geometry / expected curtain travel", padding=7)
        geom.pack(fill="x", pady=(7,0))
        rows = [
            ("Reference travel distance:", self.path_var, "mm"),
            ("Sensor C-C spacing:", self.spacing_var, "mm"),
            ("Optical aperture:", self.aperture_var, "mm"),
            ("Expected reference travel:", self.expected_travel_var, "ms"),
        ]
        for r, (label, var, unit) in enumerate(rows):
            ttk.Label(geom, text=label).grid(row=r, column=0, sticky="w", padx=(0,5), pady=2)
            ttk.Entry(geom, textvariable=var, width=9).grid(row=r, column=1, sticky="ew", pady=2)
            ttk.Label(geom, text=unit).grid(row=r, column=2, sticky="w", padx=(4,0), pady=2)

        ttk.Label(geom, text="Travel preset:").grid(row=4, column=0, sticky="w", padx=(0,5), pady=2)
        self.travel_preset_combo = ttk.Combobox(geom, values=[x[0] for x in TRAVEL_PRESETS], state="readonly")
        self.travel_preset_combo.grid(row=4, column=1, columnspan=2, sticky="ew", pady=2)
        self.travel_preset_combo.bind("<<ComboboxSelected>>", self._travel_preset_selected)

        ttk.Label(geom, text="Travel tolerance:").grid(row=5, column=0, sticky="w", padx=(0,5), pady=2)
        ttk.Combobox(geom, textvariable=self.travel_tolerance_var,
                     values=list(TRAVEL_TOLERANCE_OPTIONS), state="readonly", width=18).grid(
            row=5, column=1, columnspan=2, sticky="ew", pady=2)
        ttk.Label(geom, text="Curtain mismatch:").grid(row=6, column=0, sticky="w", padx=(0,5), pady=2)
        ttk.Combobox(geom, textvariable=self.mismatch_tolerance_var,
                     values=list(MISMATCH_TOLERANCE_OPTIONS), state="readonly", width=10).grid(
            row=6, column=1, columnspan=2, sticky="ew", pady=2)
        ttk.Label(geom, text="Exposure tolerance:").grid(row=7, column=0, sticky="w", padx=(0,5), pady=2)
        ttk.Combobox(geom, textvariable=self.tolerance_var,
                     values=list(TOLERANCE_OPTIONS), state="readonly", width=12).grid(
            row=7, column=1, columnspan=2, sticky="ew", pady=2)
        ttk.Label(geom, text="Auto uses the service-manual travel range / mismatch limit when stored in the camera CSV; "
                             "otherwise it falls back to ±5%.",
                  wraplength=330).grid(row=8, column=0, columnspan=3, sticky="w", pady=(4,0))
        ttk.Label(geom, text="Reference distance is the distance used by the stored service travel time. "
                             "Sensor C-C spacing determines the measured velocity; aperture mainly affects edge shape.",
                  wraplength=330).grid(row=9, column=0, columnspan=3, sticky="w", pady=(2,0))
        geom.columnconfigure(1, weight=1)

        # Simulator area; visibility controlled by source.
        self.simulator_frame = ttk.LabelFrame(sidebar, text="Simulator input", padding=7)
        self.simulator_frame.pack(fill="x", pady=(7,0))
        sim = self.simulator_frame
        ttk.Checkbutton(sim, text="Follow selected shutter speed", variable=self.follow_nominal_var,
                        command=self._follow_nominal_changed).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(sim, text="Actual exposure:").grid(row=1, column=0, sticky="w", pady=2)
        self.actual_combo = ttk.Combobox(sim, textvariable=self.actual_var, values=COMMON_SPEEDS)
        self.actual_combo.grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Checkbutton(sim, text="Curtains follow expected travel", variable=self.follow_travel_var,
                        command=self._follow_travel_changed).grid(row=2, column=0, columnspan=2, sticky="w", pady=(3,0))
        ttk.Label(sim, text="1st curtain:").grid(row=3, column=0, sticky="w", pady=2)
        self.sim_first_entry = ttk.Entry(sim, textvariable=self.sim_first_var)
        self.sim_first_entry.grid(row=3, column=1, sticky="ew", pady=2)
        ttk.Label(sim, text="2nd curtain:").grid(row=4, column=0, sticky="w", pady=2)
        self.sim_second_entry = ttk.Entry(sim, textvariable=self.sim_second_var)
        self.sim_second_entry.grid(row=4, column=1, sticky="ew", pady=2)
        ttk.Button(sim, text="RUN SIMULATED SHUTTER", command=self.run_simulator).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(6,0), ipady=4)
        sim.columnconfigure(1, weight=1)

        self._follow_nominal_changed()
        self._follow_travel_changed()

        # --------------------------- right side ---------------------------
        verdict = ttk.LabelFrame(main, text="Overall tolerance result", padding=9)
        verdict.pack(fill="x")
        self.indicator = tk.Canvas(verdict, width=38, height=38, highlightthickness=0)
        self.indicator.grid(row=0, column=0, rowspan=2, padx=(0,10))
        self.indicator_circle = self.indicator.create_oval(5,5,33,33,fill="#808080",outline="")
        self.pass_label = tk.Label(verdict, textvariable=self.pass_fail_var, font=("TkDefaultFont", 16, "bold"))
        self.pass_label.grid(row=0, column=1, sticky="w")
        ttk.Label(verdict, textvariable=self.tolerance_detail_var, wraplength=680).grid(row=1, column=1, sticky="w")
        ttk.Label(verdict, text="Average error:").grid(row=0, column=2, sticky="e", padx=(25,5))
        ttk.Label(verdict, textvariable=self.variance_var, font=("TkDefaultFont", 12, "bold")).grid(row=0, column=3, sticky="w")
        verdict.columnconfigure(1, weight=1)

        travel = ttk.LabelFrame(main, text="Curtain travel", padding=9)
        travel.pack(fill="x", pady=(8,0))
        self.t1_dot, self.t1_circle, self.t1_label = self._travel_row(travel, 0, "1st curtain", self.travel1_var, self.travel1_status_var)
        self.t2_dot, self.t2_circle, self.t2_label = self._travel_row(travel, 1, "2nd curtain", self.travel2_var, self.travel2_status_var)
        self.tm_dot, self.tm_circle, self.tm_label = self._travel_row(
            travel, 2, "Curtain speed mismatch", self.curtain_mismatch_var, self.curtain_mismatch_status_var)
        travel.columnconfigure(2, weight=1)

        metrics = ttk.LabelFrame(main, text="Measurements", padding=9)
        metrics.pack(fill="x", pady=(8,0))
        labels = [
            ("Average exposure", self.avg_var), ("Sensor 1 exposure", self.s1_var),
            ("Sensor 2 exposure", self.s2_var), ("Sensor 1 error", self.s1_variance_var),
            ("Sensor 2 error", self.s2_variance_var), ("Sensor mismatch", self.mismatch_var),
            ("Confidence", self.conf_var), ("Method", self.method_var),
        ]
        for i, (lab, var) in enumerate(labels):
            r, colpair = divmod(i, 2)
            c = colpair * 2
            ttk.Label(metrics, text=lab + ":").grid(row=r, column=c, sticky="e", padx=(0,5), pady=1)
            ttk.Label(metrics, textvariable=var, font=("TkDefaultFont", 10, "bold")).grid(row=r, column=c+1, sticky="w", padx=(0,18), pady=1)
        metrics.columnconfigure(1, weight=1)
        metrics.columnconfigure(3, weight=1)

        plot_frame = ttk.LabelFrame(main, text="Waveform / detected events", padding=5)
        plot_frame.pack(fill="both", expand=True, pady=(8,0))
        self.fig = Figure(figsize=(7,4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        status = ttk.Label(main, textvariable=self.status_var, relief="sunken", anchor="w", padding=(6,3))
        status.pack(fill="x", pady=(6,0))

    def _travel_row(self, frame, row, label, value_var, status_var):
        dot = tk.Canvas(frame, width=26, height=26, highlightthickness=0)
        dot.grid(row=row, column=0, rowspan=1, padx=(0,6), pady=2)
        circle = dot.create_oval(4,4,22,22,fill="#808080",outline="")
        lbl = tk.Label(frame, textvariable=status_var, width=8, font=("TkDefaultFont", 10, "bold"))
        lbl.grid(row=row, column=1, sticky="w", padx=(0,8))
        ttk.Label(frame, text=label + ":").grid(row=row, column=2, sticky="e", padx=(0,5))
        ttk.Label(frame, textvariable=value_var, font=("TkDefaultFont", 10, "bold")).grid(row=row, column=3, sticky="w")
        return dot, circle, lbl

    # ------------------------- preset / settings -------------------------
    def _apply_camera_preset(self):
        name = self.camera_var.get().strip()
        preset = CAMERA_BY_NAME.get(name)
        if preset is None:
            # allow typing a substring then hitting Return; choose exact single filtered result
            matches = [p for p in CAMERA_PRESETS if name.casefold() in p.name.casefold()]
            if len(matches) == 1:
                preset = matches[0]
                self.camera_var.set(preset.name)
            else:
                return
        self.path_var.set(f"{preset.path_mm:g}")
        if preset.travel_ms is not None:
            self.expected_travel_var.set(f"{preset.travel_ms:g}")
            if self.follow_travel_var.get():
                self.sim_first_var.set(f"{preset.travel_ms:g}")
                self.sim_second_var.set(f"{preset.travel_ms:g}")
        else:
            # Do not leave a previous camera's travel target active.  For presets
            # where the service-manual measurement geometry is unresolved, an
            # explicit user-entered target is safer than silently reusing stale data.
            self.expected_travel_var.set("")
            if self.follow_travel_var.get():
                self.sim_first_var.set("")
                self.sim_second_var.set("")
        # Default spacing follows aperture and path only if it is currently obviously stale/invalid.
        try:
            aperture = float(self.aperture_var.get())
            spacing = float(self.spacing_var.get())
            if spacing <= 0 or spacing > preset.path_mm:
                self.spacing_var.set(f"{max(0.1, preset.path_mm - aperture):g}")
        except ValueError:
            pass

        quality = {
            "factory": "documented target",
            "family": "same/family shutter value",
            "estimate": "estimated starting point",
            "unknown": "travel not documented here",
        }.get(preset.travel_quality, preset.travel_quality)
        travel = f"{preset.travel_ms:g} ms" if preset.travel_ms is not None else "not auto-set"
        if preset.service_travel_min_ms is not None or preset.service_travel_max_ms is not None:
            if preset.service_travel_min_ms is not None and preset.service_travel_max_ms is not None:
                tol_text = f"service {preset.service_travel_min_ms:g}–{preset.service_travel_max_ms:g} ms"
            elif preset.service_travel_max_ms is not None:
                tol_text = f"service ≤{preset.service_travel_max_ms:g} ms"
            else:
                tol_text = f"service ≥{preset.service_travel_min_ms:g} ms"
        elif preset.service_travel_tolerance_pct is not None:
            tol_text = f"service ±{preset.service_travel_tolerance_pct:g}%"
        else:
            tol_text = f"generic ±{GENERIC_TRAVEL_TOLERANCE_PERCENT:g}%"
        if preset.service_mismatch_max_ms is not None:
            tol_text += f", curtain Δ≤{preset.service_mismatch_max_ms:g} ms"
        extra = f" — {preset.note}" if preset.note else ""
        self.camera_info_var.set(
            f"X-sync {preset.x_sync}  |  ref {preset.path_mm:g} mm  |  travel {travel} ({quality})  |  {tol_text}{extra}"
        )
        self._clear_live_state()

    def _travel_preset_selected(self, _event=None):
        text = self.travel_preset_combo.get()
        for label, value in TRAVEL_PRESETS:
            if label == text:
                self.expected_travel_var.set(f"{value:g}")
                if self.follow_travel_var.get():
                    self.sim_first_var.set(f"{value:g}")
                    self.sim_second_var.set(f"{value:g}")
                break

    def _nominal_changed(self):
        if self.follow_nominal_var.get():
            self.actual_var.set(self.nominal_var.get())
        self._clear_live_state()

    def _follow_nominal_changed(self):
        state = "disabled" if self.follow_nominal_var.get() else "normal"
        self.actual_combo.configure(state=state)
        if self.follow_nominal_var.get():
            self.actual_var.set(self.nominal_var.get())

    def _follow_travel_changed(self):
        state = "disabled" if self.follow_travel_var.get() else "normal"
        self.sim_first_entry.configure(state=state)
        self.sim_second_entry.configure(state=state)
        if self.follow_travel_var.get():
            self.sim_first_var.set(self.expected_travel_var.get())
            self.sim_second_var.set(self.expected_travel_var.get())

    def _parse_geometry(self):
        try:
            path = float(self.path_var.get())
            spacing = float(self.spacing_var.get())
            aperture = float(self.aperture_var.get())
            travel = float(self.expected_travel_var.get())
        except ValueError as exc:
            raise ValueError(
                "Reference distance, sensor spacing, aperture and expected travel must be numeric. "
                "This camera has no automatic travel target if Expected reference travel is blank."
            ) from exc
        if path <= 0 or spacing <= 0 or aperture <= 0 or travel <= 0:
            raise ValueError("Reference distance, spacing, aperture and travel must all be positive")
        if spacing > path:
            raise ValueError("Sensor center spacing cannot exceed the reference travel distance")
        if aperture >= path:
            raise ValueError("Optical aperture width must be smaller than the reference travel distance")
        return path, spacing, aperture, travel

    # ------------------------- simulator -------------------------
    def run_simulator(self):
        try:
            nominal = parse_speed(self.nominal_var.get())
            if self.follow_nominal_var.get():
                self.actual_var.set(self.nominal_var.get())
            actual = parse_speed(self.actual_var.get())
            path, spacing, aperture, expected = self._parse_geometry()
            if self.follow_travel_var.get():
                t1 = t2 = expected
                self.sim_first_var.set(f"{expected:g}")
                self.sim_second_var.set(f"{expected:g}")
            else:
                t1, t2 = float(self.sim_first_var.get()), float(self.sim_second_var.get())
            self.status_var.set("Generating simulated stereo signal…")
            self.root.update_idletasks()
            samples = simulate_shutter(actual, t1, t2, path, spacing, aperture, self.coupling_var.get())
            self._analyze_and_display(samples, source_desc="simulator")
        except Exception as exc:
            self._set_error(str(exc), popup=True)

    # ------------------------- live audio -------------------------
    def _refresh_input_devices(self):
        values = ["Simulator"]
        self.device_map = {}
        if sd is not None:
            try:
                devices = sd.query_devices()
                for idx, d in enumerate(devices):
                    if int(d.get("max_input_channels", 0)) >= 2:
                        label = f"Live audio — {idx}: {d['name']}"
                        values.append(label)
                        self.device_map[label] = idx
            except Exception as exc:
                self.last_live_error = str(exc)
        self.source_combo["values"] = values
        if self.source_var.get() not in values:
            self.source_var.set("Simulator")
            self._source_changed()

    def _source_changed(self):
        self._stop_audio_stream()
        self._clear_live_state()
        source = self.source_var.get()
        if source == "Simulator":
            if not self.simulator_frame.winfo_ismapped():
                self.simulator_frame.pack(fill="x", pady=(7,0))
            self.status_var.set("Ready — simulator input")
        else:
            if self.simulator_frame.winfo_ismapped():
                self.simulator_frame.pack_forget()
            self._start_audio_stream()

    def _start_audio_stream(self):
        if sd is None:
            self.status_var.set("Live audio unavailable — install sounddevice")
            return
        label = self.source_var.get()
        device = self.device_map.get(label)
        if device is None:
            self.status_var.set("Selected live input device is unavailable")
            return
        try:
            self.audio_stream = sd.InputStream(
                samplerate=FS,
                device=device,
                channels=2,
                dtype="float32",
                blocksize=2048,
                callback=self._audio_callback,
            )
            self.audio_stream.start()
            self.status_var.set("LIVE — listening continuously. Fire the shutter when ready.")
        except Exception as exc:
            self.audio_stream = None
            self.status_var.set(f"Could not start live input: {exc}")

    def _stop_audio_stream(self):
        stream = self.audio_stream
        self.audio_stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _audio_callback(self, indata, frames, time_info, status):
        # PortAudio may call this from a non-Tk thread. Never touch Tk variables here.
        if status:
            self.last_live_error = str(status)
        arr = np.asarray(indata[:, :2], dtype=np.float32).copy()
        max_samples = int(12.0 * FS)
        with self.live_lock:
            self.live_chunks.append(arr)
            self.live_samples += len(arr)
            while self.live_chunks and self.live_samples > max_samples:
                old = self.live_chunks.popleft()
                self.live_samples -= len(old)

    def _clear_live_state(self):
        with self.live_lock:
            self.live_chunks.clear()
            self.live_samples = 0
        self.live_trigger_time = None
        self.live_cooldown_until = time.monotonic() + 0.1

    def _poll_live(self):
        try:
            if self.audio_stream is not None and self.live_chunks:
                self._process_live_state()
        except Exception as exc:
            # Avoid modal dialogs from the continuous loop; retain listening.
            msg = str(exc)
            if msg != self.last_live_error:
                self.last_live_error = msg
                self.status_var.set(f"LIVE — {msg}")
        finally:
            self.root.after(80, self._poll_live)

    def _process_live_state(self):
        now = time.monotonic()
        if now < self.live_cooldown_until:
            return
        with self.live_lock:
            chunks = list(self.live_chunks)
        if not chunks:
            return
        data = np.concatenate(chunks, axis=0)
        if len(data) < int(0.05 * FS):
            return

        recent = data[-min(len(data), int(0.08 * FS)):]
        baseline = data[:min(len(data), int(0.15 * FS))]
        sigma = max(robust_sigma(baseline[:,0]), robust_sigma(baseline[:,1]))
        peak = float(np.max(np.abs(recent - np.median(baseline, axis=0))))
        threshold = max(self.live_peak_floor, 8.0 * sigma)

        if self.live_trigger_time is None:
            if peak >= threshold:
                self.live_trigger_time = now
                self.status_var.set("LIVE — shutter detected; collecting closing edge…")
            return

        nominal = parse_speed(self.nominal_var.get())
        # 3-stop tolerance is intentionally generous for initial tensioning.
        wait_s = min(10.0, max(0.12, nominal * (2 ** SEARCH_STOPS) + 0.10))
        if now - self.live_trigger_time < wait_s:
            return

        # Analyze the whole retained window. The analyzer locates the shutter event.
        try:
            self._analyze_and_display(data, source_desc="live audio")
            self.status_var.set("LIVE — measurement captured. Listening for next shutter.")
        except Exception as exc:
            self.status_var.set(f"LIVE — trigger rejected ({exc}). Listening…")
        with self.live_lock:
            self.live_chunks.clear()
            self.live_samples = 0
        self.live_trigger_time = None
        self.live_cooldown_until = now + 0.25

    # ------------------------- results -------------------------
    def _analyze_and_display(self, samples, source_desc):
        nominal = parse_speed(self.nominal_var.get())
        path, spacing, aperture, expected_ms = self._parse_geometry()
        result = analyze_signal(samples, nominal, expected_ms, path, spacing, aperture,
                                self.coupling_var.get())
        self.samples = samples
        self.result = result
        self._update_results(result, nominal, path, expected_ms)
        self._update_plot(samples, result)
        self.status_var.set(
            f"Done — {source_desc}; measured {format_speed(result.exposure_average_s)}, "
            f"error {stops_from_nominal(result.exposure_average_s, nominal):+.2f} stops"
        )

    def _update_results(self, r, nominal, path_mm, expected_ms):
        self.method_var.set(r.method)
        self.avg_var.set(format_speed(r.exposure_average_s))
        self.s1_var.set(format_speed(r.exposure_sensor_1_s))
        self.s2_var.set(format_speed(r.exposure_sensor_2_s))
        avg_err = stops_from_nominal(r.exposure_average_s, nominal)
        s1_err = stops_from_nominal(r.exposure_sensor_1_s, nominal)
        s2_err = stops_from_nominal(r.exposure_sensor_2_s, nominal)
        self.variance_var.set(format_stop_error(avg_err))
        self.s1_variance_var.set(f"{s1_err:+.2f} stops")
        self.s2_variance_var.set(f"{s2_err:+.2f} stops")
        self.mismatch_var.set(f"{r.sensor_mismatch_percent:.2f}%")
        self.conf_var.set(f"{100*r.confidence:.0f}%")

        expected_s = expected_ms / 1000.0
        t1e = travel_error_percent(r.first_curtain_full_travel_s, expected_s)
        t2e = travel_error_percent(r.second_curtain_full_travel_s, expected_s)
        def travel_text(meas, err):
            if not np.isfinite(meas):
                return "—"
            speed = curtain_speed_m_s(meas, path_mm)
            return f"{meas*1000:.3f} ms  |  {speed:.2f} m/s  |  {err:+.1f}% speed"
        self.travel1_var.set(travel_text(r.first_curtain_full_travel_s, t1e))
        self.travel2_var.set(travel_text(r.second_curtain_full_travel_s, t2e))

        exp_tol = TOLERANCE_OPTIONS[self.tolerance_var.get()]

        selected_travel_tol = TRAVEL_TOLERANCE_OPTIONS[self.travel_tolerance_var.get()]
        preset = CAMERA_BY_NAME.get(self.camera_var.get().strip())
        t1_ms = r.first_curtain_full_travel_s * 1000.0
        t2_ms = r.second_curtain_full_travel_s * 1000.0

        def service_range_ok(measured_ms):
            if not np.isfinite(measured_ms):
                return False
            if preset is None:
                return None
            lo = preset.service_travel_min_ms
            hi = preset.service_travel_max_ms
            if lo is None and hi is None:
                return None
            return (lo is None or measured_ms >= lo) and (hi is None or measured_ms <= hi)

        if selected_travel_tol is None:
            range1 = service_range_ok(t1_ms)
            range2 = service_range_ok(t2_ms)
            if range1 is not None:
                t1_ok, t2_ok = range1, range2
                lo, hi = preset.service_travel_min_ms, preset.service_travel_max_ms
                if lo is not None and hi is not None:
                    travel_tol_source = f"service {lo:g}–{hi:g} ms"
                elif hi is not None:
                    travel_tol_source = f"service ≤{hi:g} ms"
                else:
                    travel_tol_source = f"service ≥{lo:g} ms"
            elif preset is not None and preset.service_travel_tolerance_pct is not None:
                travel_tol = preset.service_travel_tolerance_pct
                t1_ok = np.isfinite(t1e) and abs(t1e) <= travel_tol
                t2_ok = np.isfinite(t2e) and abs(t2e) <= travel_tol
                travel_tol_source = f"service ±{travel_tol:g}%"
            else:
                travel_tol = GENERIC_TRAVEL_TOLERANCE_PERCENT
                t1_ok = np.isfinite(t1e) and abs(t1e) <= travel_tol
                t2_ok = np.isfinite(t2e) and abs(t2e) <= travel_tol
                travel_tol_source = f"generic ±{travel_tol:g}%"
        else:
            travel_tol = selected_travel_tol
            t1_ok = np.isfinite(t1e) and abs(t1e) <= travel_tol
            t2_ok = np.isfinite(t2e) and abs(t2e) <= travel_tol
            travel_tol_source = f"manual ±{travel_tol:g}%"

        mismatch_pct = curtain_speed_mismatch_percent(
            r.first_curtain_full_travel_s, r.second_curtain_full_travel_s)
        mismatch_ms = abs(t1_ms - t2_ms) if np.isfinite(t1_ms) and np.isfinite(t2_ms) else float("nan")
        selected_mismatch_tol = MISMATCH_TOLERANCE_OPTIONS[self.mismatch_tolerance_var.get()]
        if selected_mismatch_tol is None and preset is not None and preset.service_mismatch_max_ms is not None:
            mismatch_ok = np.isfinite(mismatch_ms) and mismatch_ms <= preset.service_mismatch_max_ms
            mismatch_tol_source = f"service Δ≤{preset.service_mismatch_max_ms:g} ms"
        else:
            mismatch_tol = (GENERIC_TRAVEL_TOLERANCE_PERCENT
                            if selected_mismatch_tol is None else selected_mismatch_tol)
            mismatch_ok = np.isfinite(mismatch_pct) and mismatch_pct <= mismatch_tol
            mismatch_tol_source = (f"generic ±{mismatch_tol:g}%" if selected_mismatch_tol is None
                                   else f"manual ±{mismatch_tol:g}%")
        self.curtain_mismatch_var.set(
            f"{mismatch_pct:.2f}%  |  {mismatch_ms:.3f} ms"
            if np.isfinite(mismatch_pct) and np.isfinite(mismatch_ms) else "—")

        exposure_ok = np.isfinite(avg_err) and abs(avg_err) <= exp_tol
        overall = exposure_ok and t1_ok and t2_ok and mismatch_ok
        self._set_indicator(overall)
        self._set_small_indicator(self.t1_dot, self.t1_circle, self.t1_label, self.travel1_status_var, t1_ok)
        self._set_small_indicator(self.t2_dot, self.t2_circle, self.t2_label, self.travel2_status_var, t2_ok)
        self._set_small_indicator(self.tm_dot, self.tm_circle, self.tm_label, self.curtain_mismatch_status_var, mismatch_ok)
        self.tolerance_detail_var.set(
            f"Exposure {'passes' if exposure_ok else 'fails'} {self.tolerance_var.get()}; "
            f"curtain travel {'passes' if (t1_ok and t2_ok) else 'fails'} {travel_tol_source}; "
            f"curtain mismatch {'passes' if mismatch_ok else 'fails'} {mismatch_tol_source}."
        )

    def _set_indicator(self, passed):
        if passed is None:
            color, text, fg = "#808080", "NOT TESTED", "#404040"
        elif passed:
            color, text, fg = "#2e9b50", "PASS", "#20743b"
        else:
            color, text, fg = "#d33f3f", "FAIL", "#a32727"
        self.indicator.itemconfigure(self.indicator_circle, fill=color)
        self.pass_fail_var.set(text)
        self.pass_label.configure(fg=fg)

    @staticmethod
    def _set_small_indicator(canvas, circle, label, var, passed):
        if passed is None:
            color, text, fg = "#808080", "NOT TESTED", "#404040"
        elif passed:
            color, text, fg = "#2e9b50", "PASS", "#20743b"
        else:
            color, text, fg = "#d33f3f", "FAIL", "#a32727"
        canvas.itemconfigure(circle, fill=color)
        var.set(text)
        label.configure(fg=fg)

    def _update_plot(self, samples, r):
        self.ax.clear()
        t = np.arange(len(samples)) / FS * 1000.0
        self.ax.plot(t, samples[:,0], linewidth=0.8, label="Sensor 1")
        self.ax.plot(t, samples[:,1], linewidth=0.8, label="Sensor 2", alpha=0.8)
        for key, idx in r.event_indices.items():
            if 0 <= idx < len(samples):
                self.ax.axvline(idx / FS * 1000.0, linewidth=0.8, alpha=0.4)
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.2)
        self.ax.legend(loc="upper right")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _set_error(self, text, popup=False):
        self.status_var.set("Error — " + text)
        self._set_indicator(None)
        if popup:
            messagebox.showerror("Shutter tester", text)

    def _on_close(self):
        self._stop_audio_stream()
        self.root.destroy()


def main():
    root = tk.Tk()
    ShutterTesterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
