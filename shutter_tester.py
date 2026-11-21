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
* Reference-based integrated-light exposure, with first-order AC reconstruction checks.
* Automatic camera-specific curtain-travel tolerance where service data is known; ±5% fallback.
* Separate curtain-to-curtain speed-mismatch assessment, with service-manual limits when available.
* Selected shutter speed is a search hint, not a forced result.
* One saved summary sheet of the latest 1–20 completed tests per speed.
* ±3-stop analysis window for initial tensioning / badly misadjusted shutters.

Dependencies
------------
    pip install numpy matplotlib sounddevice

sounddevice is only required for live audio; Simulator mode works without it.
"""
from __future__ import annotations

import csv
import json
import math
import time
import threading
import tkinter as tk
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tkinter import ttk, messagebox
from typing import Optional
from uuid import uuid4

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
DEFAULT_SPACING_MM = 21.0
DEFAULT_TRAVEL_MS = 3.3
DEFAULT_AC_HZ = 35.0
SENSOR_TAU_US = 7.0
NOISE_RMS = 0.003
SEARCH_STOPS = 3.0
LIVE_EARLY_CLOSE_MIN_S = 1 / 30
DEFAULT_AC_METHOD = "AC transient edges"
INTEGRATION_METHOD = "Light integration (reference)"

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


def format_summary_measurement(seconds, nominal_s):
    stops = stops_from_nominal(seconds, nominal_s)
    if abs(stops) < 0.005:
        stops = 0.0
    return f"{seconds * 1000:.3f} ({stops:+.2f} stops)"


def passing_test_window(verdicts, count):
    """A speed is complete only when every test in a full recent window passed."""
    window = list(verdicts)[-count:]
    return len(window) == count and all(window)


def load_summary_history(path):
    """Load the current sheet without silently discarding an unreadable file."""
    try:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, {}, 5
    if not isinstance(saved, dict) or saved.get("version") != 1:
        raise ValueError("Unsupported summary file format")
    count = saved.get("count")
    speeds = saved.get("speeds")
    if type(count) is not int or not 1 <= count <= 20 or not isinstance(speeds, dict):
        raise ValueError("Invalid saved summary settings")
    history, verdicts = {}, {}
    for speed, records in speeds.items():
        nominal = float(speed)
        if not math.isfinite(nominal) or nominal <= 0 or not isinstance(records, dict):
            raise ValueError("Invalid saved shutter speed")
        exposures, passed = records.get("exposures_s"), records.get("passed")
        if (not isinstance(exposures, list) or not isinstance(passed, list)
                or len(exposures) != len(passed)
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                       for value in exposures)
                or any(type(value) is not bool for value in passed)):
            raise ValueError("Invalid saved summary measurements")
        history[nominal] = deque(exposures, maxlen=20)
        verdicts[nominal] = deque(passed, maxlen=20)
    return history, verdicts, count


def save_summary_history(path, history, verdicts, count):
    """Replace the saved sheet only after its complete contents have been written."""
    path = Path(path)
    saved = {
        "version": 1,
        "count": count,
        "speeds": {str(speed): {"exposures_s": list(values), "passed": list(verdicts[speed])}
                   for speed, values in history.items()},
    }
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(saved, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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

    # path_mm is a service reference distance, not a physical frame boundary.
    # Preserve the specified spacing even when it equals that reference.
    x1 = max(aperture_mm / 2.0, (path_mm - spacing_mm) / 2.0)
    x2 = x1 + spacing_mm

    pre = 0.020
    post = 0.050
    travel_extent = max(path_mm, x2 + aperture_mm / 2.0) / path_mm
    total = pre + exposure_s + max(travel1_ms, travel2_ms) / 1000.0 * travel_extent + post
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
    fitted_ac_cutoff_hz: tuple[float, float] | None = None
    integration_diagnostics: dict | None = None


@dataclass(frozen=True)
class IntegrationChannelReference:
    full_level: float
    polarity: int
    ac_cutoff_hz: float
    plateau_relative_rms: float
    noise_rms: float


@dataclass(frozen=True)
class IntegrationReference:
    reference_id: str
    captured_at_utc: str
    sample_rate_hz: int
    coupling: str
    channels: tuple[IntegrationChannelReference, IntegrationChannelReference]

    @classmethod
    def from_dict(cls, values):
        return cls(values["reference_id"], values["captured_at_utc"],
                   values["sample_rate_hz"], values["coupling"],
                   tuple(IntegrationChannelReference(**channel) for channel in values["channels"]))


def infer_sensor_channels(event_indices):
    """Return physical input channels in curtain-arrival order, or no order.

    Raw result fields and event keys continue to identify physical channels.
    Opening times establish one consistent order for both exposure readings.
    """
    first = event_indices.get("sensor1_open")
    second = event_indices.get("sensor2_open")
    if first is None or second is None or first == second:
        return None
    return (1, 2) if first < second else (2, 1)


def write_debug_capture(directory, raw_samples, samples, settings, result=None,
                        error=None, analysis_start_frame=0, assessment=None):
    """Save a replayable capture and readable report without pickle objects.

    Event indices in result refer to samples. raw_event_indices refer to the
    original, untrimmed raw_samples; both arrays retain the recorded polarity.
    """
    captured_at = datetime.now(timezone.utc)
    report = {
        "schema_version": 1,
        "captured_at_utc": captured_at.isoformat(),
        "outcome": ("reference" if settings.get("capture_purpose") == "full-open reference" and error is None
                    else "completed" if result is not None and error is None else "rejected"),
        "settings": settings,
        "analysis_start_frame": analysis_start_frame,
        "raw_sample_count": len(raw_samples),
        "analysis_sample_count": len(samples),
        "result": asdict(result) if result is not None else None,
        "error": error,
        "assessment": assessment,
    }
    if result is not None:
        fs = settings["sample_rate_hz"]
        nominal = settings["nominal_s"]
        path_mm = settings["reference_distance_mm"]
        channels = infer_sensor_channels(result.event_indices)
        report["sensor_order"] = {
            "basis": "opening time",
            "resolved": channels is not None,
            "first_input_channel": channels[0] if channels else None,
            "second_input_channel": channels[1] if channels else None,
            "result_field_numbering": "physical input channel",
        }
        report["raw_event_indices"] = {
            name: int(index) + analysis_start_frame
            for name, index in result.event_indices.items()
        }
        report["event_times_s"] = {
            name: index / fs for name, index in result.event_indices.items()
        }
        report["derived"] = {
            "average_error_stops": stops_from_nominal(result.exposure_average_s, nominal),
            "sensor_1_error_stops": stops_from_nominal(result.exposure_sensor_1_s, nominal),
            "sensor_2_error_stops": stops_from_nominal(result.exposure_sensor_2_s, nominal),
            "first_curtain_speed_m_s": curtain_speed_m_s(result.first_curtain_full_travel_s, path_mm),
            "second_curtain_speed_m_s": curtain_speed_m_s(result.second_curtain_full_travel_s, path_mm),
            "curtain_speed_mismatch_percent": curtain_speed_mismatch_percent(
                result.first_curtain_full_travel_s, result.second_curtain_full_travel_s),
        }

    def json_value(value):
        if isinstance(value, dict):
            return {key: json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    report_text = json.dumps(json_value(report), indent=2, ensure_ascii=False, allow_nan=False)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = captured_at.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8]
    archive = directory / f"{name}.npz"
    np.savez_compressed(archive, raw_samples=raw_samples, samples=samples,
                        metadata_json=np.asarray(report_text))
    archive.with_suffix(".json").write_text(report_text, encoding="utf-8")
    return archive


def _edge_measure_ac(channel, fs, nominal_s, *, confirm_close=False):
    """Measure the centers of the two transitions, not extrema on their tails."""
    x = smooth(channel, 5)
    n0 = min(len(x), max(64, int(0.008 * fs)))
    sigma = robust_sigma(x[:n0])
    start = min(len(x)-1, int(0.002 * fs))
    open_end = min(len(x), start + int(0.040 * fs))
    if open_end <= start + 4:
        raise ValueError("Signal window too short")

    # Find the first lobe, rather than the strongest lobe: closing may be
    # stronger than opening and both can fall inside the search window.
    x = x - np.median(x[:n0])
    threshold = max(8 * sigma, 0.008)
    hits = np.flatnonzero(np.abs(x[start:open_end]) >= threshold)
    if not len(hits):
        raise ValueError("No clear opening transient")
    first = start + int(hits[0])
    opening_sign = 1.0 if x[first] >= 0 else -1.0
    rate = np.gradient(x)
    rate_noise = robust_sigma(rate[:n0])
    opening_start = max(start, first - int(0.001 * fs))
    opening_end = min(open_end, first + max(8, int(min(0.003, nominal_s / 4) * fs)))
    open_peak = opening_start + int(np.argmax(rate[opening_start:opening_end] * opening_sign))

    low = max(4 / fs, nominal_s / (2.0 ** SEARCH_STOPS))
    high = nominal_s * (2.0 ** SEARCH_STOPS)
    close_start = max(open_peak + 2, open_peak + int(low * fs))
    close_end = min(len(x), open_peak + int(high * fs) + int(0.060 * fs))
    if confirm_close:
        # Ignore the artificial step from the smoother's padding at the buffer end.
        close_end = min(close_end, len(x) - max(4, int(0.0001 * fs)))
    if close_end <= close_start + 4:
        raise ValueError("Closing edge not yet in live buffer")

    close_peak = close_start + int(np.argmax(-opening_sign * rate[close_start:close_end]))

    def midpoint(peak, direction):
        strength = direction * rate[peak]
        if strength <= 8 * rate_noise:
            raise ValueError("No clear shutter transition")
        left = right = peak
        cutoff = max(strength * 0.10, 3 * rate_noise)
        while left > start and direction * rate[left - 1] > cutoff:
            left -= 1
        while right + 1 < len(x) and direction * rate[right + 1] > cutoff:
            right += 1
        pad = max(3, int(0.0001 * fs))
        before = float(np.median(x[max(0, left - pad):left]))
        after = float(np.median(x[right + 1:min(len(x), right + pad + 1)]))
        if direction * (after - before) <= max(6 * sigma, 1e-4):
            raise ValueError("Shutter transition is too small to measure")
        target = (before + after) / 2
        region = x[max(0, left - pad):min(len(x), right + pad + 1)]
        crossings = np.flatnonzero((direction * (region[:-1] - target) <= 0)
                                  & (direction * (region[1:] - target) >= 0))
        if not len(crossings):
            raise ValueError("Could not locate shutter transition midpoint")
        indices = crossings + max(0, left - pad)
        index = int(indices[np.argmin(np.abs(indices - peak))])
        fraction = (target - x[index]) / (x[index + 1] - x[index])
        return int(round(index + float(fraction))), abs(after - before)

    open_idx, open_height = midpoint(open_peak, opening_sign)
    close_idx, close_height = midpoint(close_peak, -opening_sign)
    if confirm_close:
        # An AC opening pulse drifts back toward baseline while the shutter
        # is still open. Require a distinct, sharp transition in the other direction.
        if (close_height < 0.12 * open_height
                or abs(rate[close_peak]) < 0.12 * abs(rate[open_peak])):
            raise ValueError("No distinct closing transient yet")
    if close_idx <= open_idx:
        raise ValueError("Closing transition must follow opening")

    measured = (close_idx - open_idx) / fs
    snr = min(open_height, close_height) / sigma
    conf = float(np.clip((snr - 3.0) / 15.0, 0.0, 1.0))
    return measured, open_idx, close_idx, conf



def _make_ac_template(exposure_s, fs, expected_travel_ms, path_mm, aperture_mm, ac_hz=None):
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
    y = one_pole_highpass(y, fs, DEFAULT_AC_HZ if ac_hz is None else ac_hz)
    y -= np.mean(y)
    return y, int(round(pre_s * fs))


def _best_normalized_match(signal, template, allow_inverted=False):
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
    if len(x) * len(t) > 2_000_000:
        fft_size = 1 << (len(x) + len(t) - 2).bit_length()
        convolution = np.fft.irfft(np.fft.rfft(x, fft_size)
                                  * np.fft.rfft(t[::-1], fft_size), fft_size)
        corr = convolution[len(t) - 1:len(x)]
    else:
        corr = np.correlate(x, t, mode="valid")
    n = len(t)
    cs = np.concatenate(([0.0], np.cumsum(x)))
    cs2 = np.concatenate(([0.0], np.cumsum(x*x)))
    sums = cs[n:] - cs[:-n]
    sums2 = cs2[n:] - cs2[:-n]
    energy = sums2 - (sums*sums)/n
    denom = np.sqrt(np.maximum(energy, 1e-12))
    score = corr / denom
    if allow_inverted:
        score = np.abs(score)
    best = int(np.argmax(score))
    return float(score[best]), best


def _estimate_ac_cutoff(channel, fs, nominal_s, crossing_s):
    """Estimate decay on a resolved illuminated plateau; short pulses use the default."""
    if ac_pulses_overlap(nominal_s, crossing_s, fs):
        return DEFAULT_AC_HZ
    try:
        duration, opened, closed, _ = _edge_measure_ac(channel, fs, nominal_s)
    except ValueError:
        return DEFAULT_AC_HZ
    margin = int(max(2 * crossing_s, 0.0005) * fs)
    lo, hi = opened + margin, closed - margin
    if duration <= 4 * crossing_s or hi - lo < int(0.001 * fs):
        return DEFAULT_AC_HZ
    baseline = float(np.median(channel[:max(8, int(0.008 * fs))]))
    values = np.abs(np.asarray(channel[lo:hi], dtype=float) - baseline)
    noise = robust_sigma(channel[:max(8, int(0.008 * fs))])
    valid = values > max(8 * noise, 1e-5)
    if np.count_nonzero(valid) < 32:
        return DEFAULT_AC_HZ
    times = np.arange(len(values))[valid] / fs
    slope = float(np.polyfit(times, np.log(values[valid]), 1)[0])
    return float(np.clip(-slope / (2 * math.pi), 0, 500))


def _template_measure_ac(channel, fs, nominal_s, expected_travel_ms, path_mm, aperture_mm, ac_hz=None):
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
    hits = np.flatnonzero(np.abs(centered) >= max(8 * robust_sigma(x[:n0]), 0.008))
    if not len(hits):
        raise ValueError("No clear shutter pulse")
    onset = int(hits[0])

    # Keep the opening even when the closing pulse is stronger and the
    # exposure is longer than the old 12 ms pre-peak crop.
    lo = max(0, onset - int(0.004 * fs))
    max_candidate_s = nominal_s * (2.0 ** SEARCH_STOPS)
    hi = min(len(x), onset + int((max_candidate_s + 0.025) * fs))
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
        crossing_s = aperture_mm * (expected_travel_ms / 1000.0) / path_mm
        if int(math.ceil((0.0015 + crossing_s + exposure_s + 0.010) * fs)) >= len(segment):
            continue
        template, open_offset = _make_ac_template(
            float(exposure_s), fs, expected_travel_ms, path_mm, aperture_mm, ac_hz
        )
        score, start = _best_normalized_match(segment, template, allow_inverted=True)
        if score > best_score:
            best_score = score
            best_s = float(exposure_s)
            best_start = lo + start
            best_open_offset = open_offset

    if best_score < 0.6:
        raise ValueError("No reliable AC pulse fit; check signal, optical aperture and expected travel")
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


def _live_analysis_start(samples, peak_floor, fs=FS):
    baseline = samples[:max(8, int(0.008 * fs))]
    centered = samples - np.median(baseline, axis=0)
    sigma = max(robust_sigma(baseline[:, 0]), robust_sigma(baseline[:, 1]))
    hits = np.flatnonzero(np.any(np.abs(centered) >= max(peak_floor, 8 * sigma), axis=1))
    if not len(hits):
        raise ValueError("No clear shutter opening in retained audio")
    return max(0, int(hits[0]) - int(0.012 * fs))


def _live_closing_ready(samples, nominal_s, path, aperture, travel_ms, coupling, fs=FS):
    """Check for a stereo close with enough recorded audio for the final analysis."""
    travel_s = travel_ms / 1000.0
    crossing_s = aperture * travel_s / path
    # Include the full integration tail and the pulse fit's 10 ms tail.
    tail = math.ceil(max(0.015, 7 * crossing_s) * fs)
    closes = []
    opens = []
    for channel in samples.T:
        if coupling.startswith("AC"):
            _, opened, closed, _ = _edge_measure_ac(channel, fs, nominal_s, confirm_close=True)
        else:
            _, opened, closed, _ = _edge_measure_dc(channel, fs, nominal_s)
            dark = channel[:int(0.008 * fs)]
            baseline = np.median(dark)
            lit_level = np.max(np.abs(channel - baseline))
            # The DC detector reports the last bright sample even while the
            # buffer ends during an exposure. Require a return to darkness.
            if abs(np.median(channel[-tail:]) - baseline) > max(
                6 * robust_sigma(dark), 0.10 * lit_level
            ):
                return False
        if not nominal_s / 2**SEARCH_STOPS <= (closed - opened + 1) / fs <= (
            nominal_s * 2**SEARCH_STOPS + 0.05
        ):
            return False
        opens.append(opened)
        closes.append(closed)
    return (len(samples) - max(closes) >= tail
            and max(abs(opens[1] - opens[0]), abs(closes[1] - closes[0])) / fs
            <= max(0.020, 3 * travel_s))


def ac_pulses_overlap(exposure_s, crossing_s, fs):
    """Leave a margin for finite aperture, audio filtering and sample resolution."""
    return exposure_s <= max(4.0 * crossing_s, 16.0 / fs)


def waveform_view_bounds(event_indices, sample_count, fs=FS):
    """Show the measured event with proportional padding, rather than idle audio."""
    indices = [i for i in event_indices.values() if 0 <= i < sample_count]
    if not indices:
        return 0, sample_count
    first, last = min(indices), max(indices)
    padding = max(int(math.ceil(0.0005 * fs)), int(math.ceil((last - first) * 0.15)))
    return max(0, first - padding), min(sample_count, last + padding + 1)


def _integration_segment(channel, opened, closed, fs, crossing_s, cutoff_hz, polarity):
    """Reconstruct local light from an initially dark, first-order AC response.

    The inverse is an explicit model assumption, not an electrical calibration.
    Retain signed baseline noise rather than clipping it into positive light.
    """
    pre = max(0.003, 4 * crossing_s)
    post = max(0.003, 6 * crossing_s)
    lo = max(0, opened - int(math.ceil(pre * fs)))
    hi = min(len(channel), closed + int(math.ceil(post * fs)))
    dark_end = opened - int(math.ceil(max(0.0004, 1.5 * crossing_s) * fs))
    dark = np.asarray(channel[lo:dark_end], dtype=float)
    tail_count = max(16, int(0.001 * fs))
    if len(dark) < 32 or hi - closed < 2 * tail_count:
        raise ValueError("Integration needs dark audio before and after the shutter")
    baseline = float(np.median(dark))
    noise = robust_sigma(dark)
    y = polarity * (np.asarray(channel[lo:hi], dtype=float) - baseline)
    light = y.copy()
    if cutoff_hz > 0:
        integral = np.concatenate(([0.0], np.cumsum((y[:-1] + y[1:]) / (2 * fs))))
        light += 2 * math.pi * cutoff_hz * integral
    return light, lo, hi, noise, tail_count


def capture_integration_reference(stereo, nominal_s, expected_travel_ms, path_mm,
                                  aperture_mm, coupling="AC coupled", fs=FS):
    """Learn each channel's full-open level and resolved AC decay from a slow shot."""
    if nominal_s < 1 / 60 - 1e-9:
        raise ValueError("Select 1/60 or slower to capture a full-open reference")
    stereo = np.asarray(stereo, dtype=float)
    if stereo.ndim != 2 or stereo.shape[1] < 2 or not np.all(np.isfinite(stereo)):
        raise ValueError("Reference needs finite stereo samples")
    if path_mm <= 0 or expected_travel_ms <= 0 or aperture_mm <= 0 or fs <= 0:
        raise ValueError("Reference geometry and sample rate must be positive")
    crossing = aperture_mm * expected_travel_ms / (1000 * path_mm)
    margin = int(math.ceil(max(0.001, 2 * crossing) * fs))
    refs = []
    for c in range(2):
        channel = stereo[:, c]
        if np.max(np.abs(channel)) >= 0.98:
            raise ValueError(f"Input channel {c + 1} clips; lower the input level")
        measure = _edge_measure_ac if coupling.startswith("AC") else _edge_measure_dc
        duration, opened, closed, _ = measure(channel, fs, nominal_s)
        a, b = opened + margin, closed - margin
        if duration < max(0.008, 8 * crossing) or b - a < int(0.004 * fs):
            raise ValueError("Reference has no resolved full-open plateau; use a slower speed")
        baseline = float(np.median(channel[:max(64, int(0.008 * fs))]))
        noise = robust_sigma(channel[:max(64, int(0.008 * fs))])
        plateau = channel[a:b] - baseline
        polarity = 1 if np.median(plateau[:max(16, int(0.002 * fs))]) >= 0 else -1
        values = polarity * plateau
        valid = values > max(10 * noise, 0.08 * float(np.max(values)))
        if np.count_nonzero(valid) < max(32, int(0.001 * fs)) or np.median(values[valid]) < 30 * noise:
            raise ValueError(f"Input channel {c + 1} reference is too weak")
        cutoff = 0.0
        if coupling.startswith("AC"):
            times = np.arange(len(values))[valid] / fs
            fit_values = values[valid]
            slope, intercept = np.polyfit(times, np.log(fit_values), 1)
            predicted = np.exp(intercept + slope * times)
            residual = np.log(fit_values) - (intercept + slope * times)
            slope_error = float(np.std(residual) / np.sqrt(np.sum((times - times.mean()) ** 2)))
            if slope > max(3 * slope_error, 0.5):
                raise ValueError("Reference illumination rises during exposure; keep the light steady")
            cutoff = max(0.0, float(-slope / (2 * math.pi)))
            if cutoff > 500:
                raise ValueError("Reference AC response decays too quickly for integration")
            fit_error = float(np.sqrt(np.mean((fit_values - predicted) ** 2)) / np.median(fit_values))
            if fit_error > 0.03:
                raise ValueError("Reference plateau does not follow a stable AC decay")
        light, lo, hi, noise, tail = _integration_segment(
            channel, opened, closed, fs, crossing, cutoff, polarity)
        full_level = float(np.median(light[a - lo:b - lo]))
        flat_error = float(np.sqrt(np.mean((light[a - lo:b - lo] - full_level) ** 2)) / full_level)
        if not np.isfinite(full_level) or full_level <= 30 * noise or flat_error > 0.04:
            raise ValueError("Reference light is not steady enough for integration")
        if abs(float(np.median(light[-tail:]))) > max(0.03 * full_level, 6 * noise):
            raise ValueError("Reference does not return to darkness after AC reconstruction")
        refs.append(IntegrationChannelReference(full_level, polarity, cutoff, flat_error, noise))
    return IntegrationReference(uuid4().hex, datetime.now(timezone.utc).isoformat(), fs,
                                coupling, tuple(refs))


def _measure_light_area(channel, opened, closed, fs, crossing_s, reference):
    if (not np.isfinite(reference.full_level) or reference.full_level <= 0
            or reference.polarity not in (-1, 1)
            or not np.isfinite(reference.ac_cutoff_hz) or reference.ac_cutoff_hz < 0):
        raise ValueError("Invalid light reference; capture a new reference")
    if np.max(np.abs(channel)) >= 0.98:
        raise ValueError("Input clips; lower the level and capture a new reference")
    light, lo, hi, noise, tail = _integration_segment(
        channel, opened, closed, fs, crossing_s, reference.ac_cutoff_hz, reference.polarity)
    level = reference.full_level
    filtered = smooth(light, max(3, int(round(0.00005 * fs))))
    peak = float(np.max(filtered))
    if peak < max(12 * noise, 0.001 * level):
        raise ValueError("Integrated shutter pulse is too weak")
    if np.min(filtered) < -max(0.03 * peak, 6 * noise):
        raise ValueError("Reconstructed light is negative; check polarity and recapture the reference")
    if peak > 1.15 * level:
        raise ValueError("Light exceeds the full-open reference; recapture the reference")
    duration = float(np.trapezoid(light, dx=1 / fs) / level)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Integrated light area is not positive; recapture the reference")
    tail_level = float(np.median(light[-tail:]))
    if abs(tail_level) > max(0.025 * level, 6 * noise):
        raise ValueError("AC reconstruction does not return to darkness; recapture the reference")
    outer_area = (np.trapezoid(light[:tail], dx=1 / fs)
                  + np.trapezoid(light[-tail:], dx=1 / fs)) / level
    if abs(outer_area) > max(0.08 * duration, 6 * noise / level * 0.002 / math.sqrt(tail)):
        raise ValueError("Integrated area depends on the window; check the AC reference and baseline")
    edge_duration = (closed - opened) / fs
    if edge_duration > max(4 * crossing_s, 0.004):
        margin = int(math.ceil(max(0.001, 2 * crossing_s) * fs))
        a, b = opened + margin - lo, closed - margin - lo
        if b > a and abs(float(np.median(light[a:b])) / level - 1) > 0.10:
            raise ValueError("Full-open light changed by over 10%; recapture the reference")
    return duration, {
        "bounds": [lo, hi], "edge_exposure_s": edge_duration,
        "normalized_area_s": duration, "full_level": level,
        "peak_fraction_of_reference": peak / level,
        "tail_fraction_of_reference": tail_level / level,
        "outer_window_area_s": float(outer_area),
        "ac_cutoff_hz": reference.ac_cutoff_hz,
    }


def analyze_signal(stereo, nominal_s, expected_travel_ms, path_mm, spacing_mm, aperture_mm,
                   coupling="AC coupled", fs=FS, ac_method=DEFAULT_AC_METHOD,
                   integration_reference=None):
    stereo = np.asarray(stereo)
    if stereo.ndim != 2 or stereo.shape[1] < 2:
        raise ValueError("Expected stereo input with two channels")
    stereo = stereo[:, :2]
    integrate = ac_method == INTEGRATION_METHOD
    if integrate:
        if integration_reference is None:
            raise ValueError("Capture a full-open reference at 1/60 or slower first")
        if isinstance(integration_reference, dict):
            integration_reference = IntegrationReference.from_dict(integration_reference)
        if (integration_reference.sample_rate_hz != fs or integration_reference.coupling != coupling
                or len(integration_reference.channels) != 2):
            raise ValueError("Reference sample rate or signal mode changed; capture a new reference")

    # A finite optical aperture smears very fast shutter pulses. When the
    # selected speed is short enough for opening/closing AC transients to
    # overlap, fit the entire expected pulse shape. Otherwise use edges.
    expected_travel_s = expected_travel_ms / 1000.0
    sensor_crossing_s = aperture_mm * expected_travel_s / path_mm
    is_ac = coupling.startswith("AC")
    use_template = is_ac and (
        ac_method == "AC pulse fit"
        or (ac_method == "Auto" and ac_pulses_overlap(nominal_s, sensor_crossing_s, fs))
    )

    # A selected slow speed may actually fire fast. In Auto mode, use the
    # observed duration too, and apply the same method to both sensors.
    edge_results = None
    if is_ac and not use_template:
        edge_results = [_edge_measure_ac(stereo[:, c], fs, nominal_s) for c in range(2)]
        if ac_method == "Auto" and any(
            ac_pulses_overlap(result[0], sensor_crossing_s, fs) for result in edge_results
        ):
            use_template = True

    measurements, opens, closes, confs = [], [], [], []
    fitted_cutoffs = []
    integration_details = []
    for c in range(2):
        if use_template:
            cutoff = _estimate_ac_cutoff(stereo[:, c], fs, nominal_s, sensor_crossing_s)
            fitted_cutoffs.append(cutoff)
            exp, oi, ci, conf = _template_measure_ac(
                stereo[:, c], fs, nominal_s, expected_travel_ms, path_mm, aperture_mm, cutoff
            )
        elif coupling.startswith("AC"):
            exp, oi, ci, conf = edge_results[c]
        else:
            exp, oi, ci, conf = _edge_measure_dc(stereo[:, c], fs, nominal_s)
        if integrate:
            exp, details = _measure_light_area(stereo[:, c], oi, ci, fs, sensor_crossing_s,
                                              integration_reference.channels[c])
            integration_details.append(details)
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
        method=(INTEGRATION_METHOD if integrate else "AC template fit" if use_template else
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
        fitted_ac_cutoff_hz=tuple(fitted_cutoffs) if fitted_cutoffs else None,
        integration_diagnostics=({"reference_id": integration_reference.reference_id,
                                  "channels": integration_details,
                                  "timing_basis": "Edge markers measure curtain travel; exposure is normalized light area",
                                  "reference_condition": "Fixed illumination, gain and sensor position required",
                                  "ac_model": "first-order" if is_ac else "direct"}
                                 if integrate else None),
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
    def __init__(self, root, summary_path=None):
        self.root = root
        root.title("Two-Point Shutter Tester v12")
        root.geometry("1320x820")
        root.minsize(1050, 680)

        self.samples = None
        self.result = None
        self.result_context = None
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
        self.spacing_var = tk.StringVar(value=f"{DEFAULT_SPACING_MM:g}")
        self.aperture_var = tk.StringVar(value="2")
        self.expected_travel_var = tk.StringVar(value="3.3")
        self.travel_tolerance_var = tk.StringVar(value="Auto (service / ±5%)")
        self.mismatch_tolerance_var = tk.StringVar(value="Auto (service / ±5%)")
        self.coupling_var = tk.StringVar(value="AC coupled")
        self.ac_method_var = tk.StringVar(value=DEFAULT_AC_METHOD)
        self.debug_mode_var = tk.BooleanVar(value=False)
        self.debug_status_var = tk.StringVar(value="Debug recording off")
        self.debug_directory = Path(__file__).resolve().parent / "debug_captures"
        self.integration_reference = None
        self.integration_reference_context = None
        self.reference_pending = False
        self.reference_previous_nominal = None
        self.reference_status_var = tk.StringVar(value="No reference captured")
        self.measurement_note_var = tk.StringVar(value="Exposure uses transition timing.")

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
        self.s1_channel_var = tk.StringVar(value="Waiting for timing")
        self.s2_channel_var = tk.StringVar(value="Waiting for timing")
        self.s1_duration_var = tk.StringVar(value="—")
        self.s2_duration_var = tk.StringVar(value="—")
        self.avg_duration_var = tk.StringVar(value="—")
        self.avg_variance_var = tk.StringVar(value="—")
        self.first_sensor_title_var = tk.StringVar(value="First sensor")
        self.second_sensor_title_var = tk.StringVar(value="Second sensor")
        self.sensor_order_var = tk.StringVar(value="Sensor order will be inferred from opening times.")
        self.exposure_status_vars = {
            name: tk.StringVar(value="NOT TESTED") for name in ("first", "average", "second")
        }
        self.exposure_status_labels = {}
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

        self.test_history = {}
        self.test_pass_history = {}
        self.summary_path = (Path(summary_path) if summary_path is not None else
                             Path(__file__).resolve().with_name("shutter_summary.json"))
        self.summary_count_var = tk.StringVar(value="5")
        self.summary_storage_status_var = tk.StringVar()
        self.summary_save_blocked = False
        self._restore_summary()
        self.speed_buttons = {}
        self._build_ui()
        self.summary_count_var.trace_add("write", self._summary_count_changed)
        for var in (self.source_var, self.camera_var, self.coupling_var, self.path_var,
                    self.spacing_var, self.aperture_var, self.expected_travel_var):
            var.trace_add("write", self._reference_settings_changed)
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

        notebook = ttk.Notebook(outer)
        notebook.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        main = ttk.Frame(notebook, padding=6)
        notebook.add(main, text="Current test")
        summary = ttk.Frame(notebook, padding=18)
        notebook.add(summary, text="Shutter speed summary")
        self._build_summary_panel(summary)

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
        ttk.Label(src, text="Exposure method:").grid(row=3, column=0, sticky="w", padx=(0, 6))
        ac_method = ttk.Combobox(
            src, textvariable=self.ac_method_var,
            values=[INTEGRATION_METHOD, "AC transient edges", "AC pulse fit", "Auto"], state="readonly",
        )
        ac_method.grid(row=3, column=1, columnspan=2, sticky="ew", pady=2)
        ac_method.bind("<<ComboboxSelected>>", self._analysis_method_changed)
        ttk.Checkbutton(
            src, text="Debug mode: save every test", variable=self.debug_mode_var,
            command=lambda: self.debug_status_var.set(
                f"Saving to {self.debug_directory}" if self.debug_mode_var.get()
                else "Debug recording off"
            ),
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(5, 0))
        ttk.Label(src, textvariable=self.debug_status_var, wraplength=330).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=2
        )
        src.columnconfigure(1, weight=1)

        reference_box = ttk.LabelFrame(sidebar, text="Light integration reference", padding=7)
        reference_box.pack(fill="x", pady=(7, 0))
        self.reference_capture_button = ttk.Button(
            reference_box, text="Capture reference from next shot", command=self._request_reference_capture)
        self.reference_capture_button.pack(fill="x")
        ttk.Button(reference_box, text="Clear reference", command=self._clear_integration_reference).pack(
            anchor="e", pady=(4, 0))
        ttk.Label(reference_box, textvariable=self.reference_status_var, wraplength=330).pack(
            anchor="w", pady=(4, 0))
        ttk.Label(reference_box,
                  text="Use 1/60 or slower for the reference. Keep the light, input gain and sensor position fixed. "
                       "Recapture after changes. AC integration assumes a first-order response.",
                  wraplength=330).pack(anchor="w", pady=(4, 0))

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
                               indicatoron=False, width=8, padx=1, pady=3, relief="raised",
                               command=self._nominal_changed)
            r, c = divmod(i, 4)
            b.grid(row=r, column=c, sticky="ew", padx=1, pady=1)
            speeds.columnconfigure(c, weight=1)
            self.speed_buttons[speed] = b

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
        exposure_tolerance = ttk.Combobox(geom, textvariable=self.tolerance_var,
                                        values=list(TOLERANCE_OPTIONS), state="readonly", width=12)
        exposure_tolerance.grid(row=7, column=1, columnspan=2, sticky="ew", pady=2)
        exposure_tolerance.bind("<<ComboboxSelected>>", self._refresh_result_assessment)
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

        metrics = ttk.Frame(main)
        metrics.pack(fill="x", pady=(8,0))
        for column, title_var, channel_var, exposure_var, duration_var, error_var in (
            (0, self.first_sensor_title_var, self.s1_channel_var, self.s1_var,
             self.s1_duration_var, self.s1_variance_var),
            (1, None, None, self.avg_var, self.avg_duration_var, self.avg_variance_var),
            (2, self.second_sensor_title_var, self.s2_channel_var, self.s2_var,
             self.s2_duration_var, self.s2_variance_var),
        ):
            metrics.columnconfigure(column, weight=1, uniform="measurement_boxes")
            box = ttk.LabelFrame(metrics, text="Average" if title_var is None else "", padding=10)
            if title_var is not None:
                box.configure(labelwidget=ttk.Label(box, textvariable=title_var))
            box.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 4, 0 if column == 2 else 4))
            if channel_var is not None:
                ttk.Label(box, textvariable=channel_var, anchor="center").pack(fill="x")
            else:
                ttk.Label(box, text="Both sensors", anchor="center").pack(fill="x")
            ttk.Label(box, textvariable=exposure_var, font=("TkDefaultFont", 18, "bold"),
                      anchor="center").pack(fill="x", pady=(6, 2))
            ttk.Label(box, textvariable=duration_var, anchor="center").pack(fill="x")
            ttk.Label(box, textvariable=error_var, font=("TkDefaultFont", 11, "bold"),
                      anchor="center").pack(fill="x", pady=(4, 0))
            status_name = ("first", "average", "second")[column]
            status_label = ttk.Label(
                box, textvariable=self.exposure_status_vars[status_name],
                font=("TkDefaultFont", 12, "bold"), foreground="#808080", anchor="center",
            )
            status_label.pack(fill="x", pady=(6, 0))
            self.exposure_status_labels[status_name] = status_label
        ttk.Label(metrics, textvariable=self.sensor_order_var, wraplength=740).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(5, 3)
        )
        details = ttk.Frame(metrics)
        details.grid(row=2, column=0, columnspan=3, sticky="ew")
        for column, (label, var) in enumerate((
            ("Sensor mismatch", self.mismatch_var),
            ("Signal quality", self.conf_var), ("Method", self.method_var),
        )):
            ttk.Label(details, text=label + ":").grid(row=0, column=column * 2, sticky="w", padx=(0, 5))
            ttk.Label(details, textvariable=var, font=("TkDefaultFont", 10, "bold")).grid(
                row=0, column=column * 2 + 1, sticky="w", padx=(0, 15)
            )
            details.columnconfigure(column * 2 + 1, weight=1)
        ttk.Label(metrics, textvariable=self.measurement_note_var, wraplength=740).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))

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

    def _reference_context(self):
        return tuple(var.get() for var in (
            self.source_var, self.camera_var, self.coupling_var, self.path_var,
            self.spacing_var, self.aperture_var, self.expected_travel_var))

    def _reference_settings_changed(self, *_args):
        if self.integration_reference is not None or self.reference_pending:
            self._clear_integration_reference()
            self.reference_status_var.set("Settings changed; capture a new reference")

    def _analysis_method_changed(self, _event=None):
        self._clear_live_state()
        self._refresh_summary()

    def _request_reference_capture(self):
        if self.reference_pending:
            self._clear_integration_reference()
            return
        self.integration_reference = None
        self.integration_reference_context = None
        self.reference_previous_nominal = self.nominal_var.get()
        self.reference_pending = True
        if parse_speed(self.nominal_var.get()) < 1 / 60:
            self.nominal_var.set("1/60")
        self._nominal_changed()
        self.reference_capture_button.configure(text="Cancel reference capture")
        self.reference_status_var.set(
            f"Set the camera to {self.nominal_var.get()} and fire once. The next shot saves the reference.")
        self._refresh_summary()
        if self.source_var.get() == "Simulator":
            self.run_simulator()

    def _clear_integration_reference(self):
        self.integration_reference = None
        self.integration_reference_context = None
        self.reference_pending = False
        if self.reference_previous_nominal is not None:
            self.nominal_var.set(self.reference_previous_nominal)
            self.reference_previous_nominal = None
            self._nominal_changed()
        self.reference_capture_button.configure(text="Capture reference from next shot")
        self.reference_status_var.set("No reference captured")
        self._clear_live_state()
        self._refresh_summary()

    def _restore_summary(self):
        try:
            self.test_history, self.test_pass_history, count = load_summary_history(self.summary_path)
        except (OSError, ValueError, TypeError) as exc:
            self.summary_save_blocked = True
            self.summary_storage_status_var.set(
                f"Saved summary could not be loaded: {exc}. File preserved; Clear summary starts a new sheet.")
        else:
            self.summary_count_var.set(str(count))

    def _save_summary(self):
        if self.summary_save_blocked:
            return
        try:
            save_summary_history(self.summary_path, self.test_history, self.test_pass_history,
                                 int(self.summary_count_var.get()))
        except (OSError, ValueError, TypeError) as exc:
            self.summary_storage_status_var.set(f"Summary kept in memory, but could not be saved: {exc}")
        else:
            self.summary_storage_status_var.set("")

    def _summary_count_changed(self, *_args):
        self._save_summary()
        self._refresh_summary()

    def _build_summary_panel(self, panel):
        ttk.Label(
            panel, text="Shutter speed test summary",
            font=("TkDefaultFont", 20, "bold"),
        ).pack(anchor="w")
        self.summary_source_var = tk.StringVar()
        ttk.Label(panel, textvariable=self.summary_source_var).pack(
            anchor="w", pady=(6, 4)
        )
        options = ttk.Frame(panel)
        options.pack(fill="x", pady=(0, 6))
        ttk.Label(options, text="Latest tests per speed:").pack(side="left")
        count_selector = ttk.Combobox(
            options, textvariable=self.summary_count_var,
            values=[str(count) for count in range(1, 21)],
            state="readonly", width=4,
        )
        count_selector.pack(side="left", padx=(8, 0))
        ttk.Label(
            panel,
            text="Last 20 completed tests retained per speed; exposure averaged across both sensors.",
        ).pack(anchor="w", pady=(0, 16))

        style = ttk.Style(self.root)
        style.configure("Summary.Treeview", rowheight=30, font=("TkDefaultFont", 11))
        style.configure("Summary.Treeview.Heading", font=("TkDefaultFont", 10, "bold"))
        self.summary_table = ttk.Treeview(
            panel, columns=("speed", "count", "min", "average", "max"),
            show="headings", height=len(COMMON_SPEEDS), style="Summary.Treeview",
            selectmode="none",
        )
        for column, title, width in (
            ("speed", "Marked speed", 110), ("count", "Tests", 65),
            ("min", "Min (ms / stops)", 200), ("average", "Average (ms / stops)", 200),
            ("max", "Max (ms / stops)", 200),
        ):
            self.summary_table.heading(column, text=title)
            self.summary_table.column(column, width=width, minwidth=60, anchor="center")
        self.summary_table.tag_configure("alternate", background="#eef3f8")
        self.summary_table.pack(fill="both", expand=True)
        footer = ttk.Frame(panel)
        footer.pack(fill="x", pady=(12, 0))
        ttk.Label(
            footer, text="Saved until Clear summary, including after restarting. Min / max refer to exposure duration."
        ).pack(side="left")
        ttk.Button(footer, text="Clear summary", command=self._clear_summary).pack(side="right")
        ttk.Label(panel, textvariable=self.summary_storage_status_var, foreground="#a32727",
                  wraplength=700).pack(anchor="w", pady=(4, 0))
        ttk.Label(panel, text="Signed stops vs marked speed: + longer exposure, - shorter exposure.").pack(
            anchor="w", pady=(4, 0)
        )
        ttk.Label(panel, text="✓ on a speed button means the selected number of latest tests all recorded overall PASS.").pack(
            anchor="w", pady=(4, 0)
        )
        self._refresh_summary()

    def _refresh_summary(self):
        count = int(self.summary_count_var.get())
        self.summary_source_var.set("Current sheet — all recorded tests")
        history = self.test_history
        self.summary_table.delete(*self.summary_table.get_children())
        tested_speeds = [speed for speed in COMMON_SPEEDS if history.get(parse_speed(speed))]
        for index, speed in enumerate(tested_speeds):
            values = list(history.get(parse_speed(speed), ()))[-count:]
            stats = tuple(format_summary_measurement(value, parse_speed(speed)) for value in (
                min(values), sum(values) / len(values), max(values)
            ))
            self.summary_table.insert(
                "", "end", values=(speed, f"{len(values)}/{count}", *stats),
                tags=("alternate",) if index % 2 else (),
            )
        self._update_speed_checkmarks()

    def _update_speed_checkmarks(self):
        count = int(self.summary_count_var.get())
        history = self.test_pass_history
        for speed, button in self.speed_buttons.items():
            passed = passing_test_window(history.get(parse_speed(speed), ()), count)
            button.configure(text=f"✓ {speed}" if passed else speed)

    def _record_summary(self, result: AnalysisResult, nominal_s: float):
        exposure = result.exposure_average_s
        if not np.isfinite(exposure) or exposure <= 0:
            return
        self.test_history.setdefault(nominal_s, deque(maxlen=20)).append(exposure)
        self.test_pass_history.setdefault(nominal_s, deque(maxlen=20)).append(self.pass_fail_var.get() == "PASS")
        self._save_summary()
        self._refresh_summary()

    def _clear_summary(self):
        self.test_history.clear()
        self.test_pass_history.clear()
        self.summary_save_blocked = False
        self._save_summary()
        self._refresh_summary()

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
        self._refresh_summary()
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
        timed_out = now - self.live_trigger_time >= wait_s
        check_close = nominal >= LIVE_EARLY_CLOSE_MIN_S
        ready = False
        if check_close:
            try:
                path, _, aperture, travel_ms = self._parse_geometry()
                start = _live_analysis_start(data, self.live_peak_floor)
                ready = _live_closing_ready(data[start:], nominal, path, aperture, travel_ms,
                                            self.coupling_var.get())
            except ValueError:
                pass  # A partial capture will be checked again as more audio arrives.
        if not ready and not timed_out:
            return

        try:
            capturing_reference = self.reference_pending
            if check_close and not ready:
                self._analyze_and_display(data, source_desc="live audio", closing_timeout=True)
            else:
                self._analyze_and_display(data, source_desc="live audio")
            if not capturing_reference:
                self.status_var.set("LIVE — measurement captured. Listening for next shutter.")
        except Exception as exc:
            self.status_var.set(f"LIVE — trigger rejected ({exc}). Listening…")
        with self.live_lock:
            self.live_chunks.clear()
            self.live_samples = 0
        self.live_trigger_time = None
        self.live_cooldown_until = now + 0.25

    # ------------------------- results -------------------------
    def _debug_settings(self, source_desc):
        return {
            "sample_rate_hz": FS,
            "source_kind": source_desc,
            "input_source": self.source_var.get(),
            "camera": self.camera_var.get(),
            "nominal_speed": self.nominal_var.get(),
            "reference_distance_mm_text": self.path_var.get(),
            "sensor_spacing_mm_text": self.spacing_var.get(),
            "optical_aperture_mm_text": self.aperture_var.get(),
            "expected_travel_ms_text": self.expected_travel_var.get(),
            "coupling": self.coupling_var.get(),
            "ac_method": self.ac_method_var.get(),
            "exposure_tolerance": self.tolerance_var.get(),
            "travel_tolerance": self.travel_tolerance_var.get(),
            "mismatch_tolerance": self.mismatch_tolerance_var.get(),
            "live_trigger_floor": self.live_peak_floor,
            "template_ac_cutoff_hz": DEFAULT_AC_HZ,
            "template_sensor_tau_us": SENSOR_TAU_US,
            "search_stops": SEARCH_STOPS,
            "capture_purpose": ("full-open reference" if getattr(self, "reference_pending", False)
                                else "measurement"),
            "integration_reference": (asdict(self.integration_reference)
                                      if getattr(self, "integration_reference", None) is not None else None),
            "integration_reference_context": getattr(self, "integration_reference_context", None),
            "simulator": {
                "actual_speed": self.actual_var.get(),
                "first_travel_ms": self.sim_first_var.get(),
                "second_travel_ms": self.sim_second_var.get(),
            } if source_desc == "simulator" else None,
        }

    def _clear_measurement_values(self, status):
        for var in (self.s1_var, self.s2_var, self.avg_var, self.s1_duration_var,
                    self.s2_duration_var, self.avg_duration_var, self.s1_variance_var,
                    self.s2_variance_var, self.avg_variance_var, self.variance_var,
                    self.travel1_var, self.travel2_var, self.curtain_mismatch_var,
                    self.mismatch_var, self.conf_var):
            var.set("—")
        self._set_indicator(None)
        for name, var in self.exposure_status_vars.items():
            var.set(status)
            self.exposure_status_labels[name].configure(foreground="#808080")
        self._set_small_indicator(self.t1_dot, self.t1_circle, self.t1_label, self.travel1_status_var, None)
        self._set_small_indicator(self.t2_dot, self.t2_circle, self.t2_label, self.travel2_status_var, None)
        self._set_small_indicator(self.tm_dot, self.tm_circle, self.tm_label, self.curtain_mismatch_status_var, None)

    def _accept_integration_reference(self, reference, samples, nominal, path, spacing, aperture, expected_ms):
        markers = analyze_signal(samples, nominal, expected_ms, path, spacing, aperture,
                                 self.coupling_var.get(), ac_method=DEFAULT_AC_METHOD)
        self.integration_reference = reference
        self.integration_reference_context = self._reference_context()
        self.reference_pending = False
        self.reference_capture_button.configure(text="Capture reference from next shot")
        self.reference_status_var.set("Reference ready. Keep light, gain and sensor position fixed.")
        self.ac_method_var.set(INTEGRATION_METHOD)
        if self.reference_previous_nominal is not None:
            self.nominal_var.set(self.reference_previous_nominal)
            self.reference_previous_nominal = None
            self._nominal_changed()
        self.samples = samples
        self.result = None
        self.result_context = None
        self._update_plot(samples, markers)
        self.method_var.set("Full-open reference captured")
        self._clear_measurement_values("REFERENCE")
        self.tolerance_detail_var.set("Reference saved; this shot is not included in the test summary.")
        self.measurement_note_var.set("Reference-based integration is selected. Select a test speed and fire.")
        self.status_var.set(f"Reference saved. Set the camera to {self.nominal_var.get()} and fire; listening continues.")
        self._refresh_summary()

    def _analyze_and_display(self, samples, source_desc, *, closing_timeout=False):
        debug = self.debug_mode_var.get()
        settings = self._debug_settings(source_desc) if debug else None
        raw_samples = samples
        start = 0
        result = None
        error = None
        assessment = None
        try:
            nominal = parse_speed(self.nominal_var.get())
            path, spacing, aperture, expected_ms = self._parse_geometry()
            if debug:
                settings.update(nominal_s=nominal, reference_distance_mm=path,
                                sensor_spacing_mm=spacing, optical_aperture_mm=aperture,
                                expected_travel_ms=expected_ms)
            if source_desc == "live audio":
                # Keep the original buffer for debug replay before trimming idle audio.
                start = _live_analysis_start(samples, self.live_peak_floor)
                samples = samples[start:]
            if closing_timeout:
                raise ValueError("Timed out waiting for both sensors to see the shutter close")
            if getattr(self, "reference_pending", False):
                reference = capture_integration_reference(
                    samples, nominal, expected_ms, path, aperture, self.coupling_var.get())
                self._accept_integration_reference(reference, samples, nominal, path, spacing, aperture, expected_ms)
                if debug:
                    settings["integration_reference"] = asdict(reference)
                    settings["integration_reference_context"] = self.integration_reference_context
                    assessment = {"overall": "REFERENCE", "detail": "Full-open reference; not a test result"}
                return
            reference = None
            if self.ac_method_var.get() == INTEGRATION_METHOD:
                reference = getattr(self, "integration_reference", None)
                if reference is not None and self.integration_reference_context != self._reference_context():
                    raise ValueError("Reference settings changed; capture a new reference")
            result = analyze_signal(samples, nominal, expected_ms, path, spacing, aperture,
                                    self.coupling_var.get(), ac_method=self.ac_method_var.get(),
                                    integration_reference=reference)
            self.samples = samples
            self.result = result
            self.result_context = (nominal, path, expected_ms)
            self._update_results(result, nominal, path, expected_ms)
            self._update_plot(samples, result)
            self._record_summary(result, nominal)
            self.status_var.set(
                f"Done — {source_desc}; measured {format_speed(result.exposure_average_s)}, "
                f"error {stops_from_nominal(result.exposure_average_s, nominal):+.2f} stops"
            )
            if debug:
                assessment = {"overall": self.pass_fail_var.get(),
                              "detail": self.tolerance_detail_var.get()}
        except Exception as exc:
            error = str(exc)
            if self.ac_method_var.get() == INTEGRATION_METHOD or getattr(self, "reference_pending", False):
                self.result = None
                self.result_context = None
                self._clear_measurement_values("REJECTED")
                self.method_var.set("Integration / reference rejected")
                self.tolerance_detail_var.set(error)
                self.measurement_note_var.set("No exposure result was recorded for this capture.")
                self.ax.clear()
                self.ax.text(0.5, 0.5, "Reference / signal check failed\n" + error,
                             ha="center", va="center", wrap=True, transform=self.ax.transAxes)
                self.canvas.draw_idle()
            raise
        finally:
            if debug:
                try:
                    archive = write_debug_capture(
                        self.debug_directory, raw_samples, samples, settings, result,
                        error, start, assessment,
                    )
                    self.debug_status_var.set(f"Saved: {archive.name}")
                except Exception as exc:
                    # Recording errors must not discard an otherwise valid measurement.
                    self.debug_status_var.set(f"Debug save failed: {exc}")

    def _refresh_result_assessment(self, _event=None):
        if self.result is not None and self.result_context is not None:
            self._update_results(self.result, *self.result_context)

    def _update_results(self, r, nominal, path_mm, expected_ms):
        self.method_var.set(r.method)
        if hasattr(self, "measurement_note_var"):
            self.measurement_note_var.set(
                "Exposure is integrated light / full-open reference. Plot markers show curtain edge timing."
                if r.integration_diagnostics else "Exposure uses transition timing; signal quality is not timing accuracy.")
        self.avg_var.set(format_speed(r.exposure_average_s))
        channels = infer_sensor_channels(r.event_indices)
        first_channel, second_channel = channels or (1, 2)
        exposures = {1: r.exposure_sensor_1_s, 2: r.exposure_sensor_2_s}
        first_exposure, second_exposure = exposures[first_channel], exposures[second_channel]
        self.s1_var.set(format_speed(first_exposure))
        self.s2_var.set(format_speed(second_exposure))
        self.s1_channel_var.set(f"Input channel {first_channel}")
        self.s2_channel_var.set(f"Input channel {second_channel}")
        self.first_sensor_title_var.set("First sensor" if channels else "Sensor A")
        self.second_sensor_title_var.set("Second sensor" if channels else "Sensor B")
        self.sensor_order_var.set(
            "First/second follow the detected opening times."
            if channels else "Opening times are indistinguishable; sensor order is unresolved."
        )
        self.s1_duration_var.set(f"{first_exposure * 1000:.4f} ms")
        self.s2_duration_var.set(f"{second_exposure * 1000:.4f} ms")
        self.avg_duration_var.set(f"{r.exposure_average_s * 1000:.4f} ms")
        avg_err = stops_from_nominal(r.exposure_average_s, nominal)
        s1_err = stops_from_nominal(first_exposure, nominal)
        s2_err = stops_from_nominal(second_exposure, nominal)
        self.variance_var.set(format_stop_error(avg_err))
        self.avg_variance_var.set(f"{avg_err:+.2f} stops")
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
        exposure_passes = {}
        for name, error in (("first", s1_err), ("average", avg_err), ("second", s2_err)):
            passed = bool(np.isfinite(error) and abs(error) <= exp_tol)
            exposure_passes[name] = passed
            self.exposure_status_vars[name].set("PASS" if passed else "FAIL")
            self.exposure_status_labels[name].configure(
                foreground="#20743b" if passed else "#a32727"
            )

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

        exposure_ok = all(exposure_passes.values())
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
        lo, hi = waveform_view_bounds(r.event_indices, len(samples))
        t = np.arange(lo, hi) / FS * 1000.0
        channels = infer_sensor_channels(r.event_indices)
        order = channels or (1, 2)
        for position, channel in enumerate(order):
            name = ("First sensor", "Second sensor")[position] if channels else f"Sensor {'AB'[position]}"
            self.ax.plot(t, samples[lo:hi, channel - 1], linewidth=0.8,
                         color=f"C{position}", label=f"{name} (input channel {channel})")
        estimated = "fit" in r.method.lower()
        for key, idx in r.event_indices.items():
            if 0 <= idx < len(samples):
                channel = 1 if key.startswith("sensor1") else 2
                sensor = order.index(channel) + 1
                opening = key.endswith("open")
                label = f"{'S' + str(sensor) if channels else 'CH' + str(channel)} {'open' if opening else 'close'}"
                if estimated:
                    label += " (fit)"
                elif r.integration_diagnostics:
                    label += " (edge)"
                self.ax.axvline(
                    idx / FS * 1000.0, linewidth=1.0, alpha=0.65,
                    color=f"C{sensor - 1}", linestyle="--" if opening else ":",
                    label=label,
                )
        if hi - lo > 1:
            self.ax.set_xlim(t[0], t[-1])
        note = ("\nExposure is light area; markers are edge timings for curtain travel"
                if r.integration_diagnostics else " — opening/closing times estimated from pulse shape" if estimated else "")
        self.ax.set_title(f"{r.method}{note}", fontsize=10 if r.integration_diagnostics else 12)
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.2)
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _set_error(self, text, popup=False):
        self.status_var.set("Error — " + text)
        self._set_indicator(None)
        if popup:
            messagebox.showerror("Shutter tester", text)

    def _on_close(self):
        self._save_summary()
        self._stop_audio_stream()
        self.root.destroy()


def main():
    root = tk.Tk()
    ShutterTesterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
