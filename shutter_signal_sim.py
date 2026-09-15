#!/usr/bin/env python3
"""
Two-point focal-plane shutter sensor simulator.

The output is stereo:
    channel 1 = sensor near the start of curtain travel
    channel 2 = sensor near the far edge

The sensor is NOT modeled as a point. Each sensor covers a finite width of
the film gate (2 mm by default), and its output is the average illumination
across that area. This naturally reproduces the short, reduced-height
"bump" seen when a fast shutter slit is narrower than the sensor.

Requires:
    pip install numpy

Optional for --play:
    pip install sounddevice
"""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import numpy as np


def parse_speed(text: str) -> float:
    """Convert '1/1000', '0.001', etc. to exposure time in seconds."""
    text = text.strip()
    if "/" in text:
        a, b = text.split("/", 1)
        value = float(a) / float(b)
    else:
        value = float(text)

    if value <= 0:
        raise argparse.ArgumentTypeError("Shutter speed/exposure must be > 0")

    return value


def one_pole_lowpass(x: np.ndarray, fs: float, tau_s: float) -> np.ndarray:
    """Optional sensor/electronics response model."""
    if tau_s <= 0:
        return x.copy()

    alpha = 1.0 - math.exp(-1.0 / (fs * tau_s))
    y = np.empty_like(x)
    y[0] = alpha * x[0]

    for i in range(1, len(x)):
        y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])

    return y


def one_pole_highpass(x: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    """Optional approximation of AC coupling in an audio input."""
    if cutoff_hz <= 0:
        return x.copy()

    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / fs
    alpha = rc / (rc + dt)

    y = np.zeros_like(x)
    for i in range(1, len(x)):
        y[i] = alpha * (y[i - 1] + x[i] - x[i - 1])

    return y


def sensor_signal(
    t: np.ndarray,
    x_start_mm: float,
    x_end_mm: float,
    v_first_mm_s: float,
    v_second_mm_s: float,
    exposure_s: float,
    spatial_samples: int = 128,
) -> np.ndarray:
    """
    Calculate average illumination across a finite-width optical sensor.

    At each tiny point x:

        opening time = x / first-curtain velocity
        closing time = exposure + x / second-curtain velocity

    If both curtains have the same velocity, every point receives exactly
    the nominal exposure time.

    Spatial averaging is what turns an extremely narrow moving slit into
    a reduced-height bump instead of an ideal square pulse.
    """
    if spatial_samples < 4:
        raise ValueError("spatial_samples must be >= 4")

    width_mm = x_end_mm - x_start_mm
    dx_mm = width_mm / spatial_samples

    # Sample the center of each small spatial segment.
    xs = x_start_mm + (np.arange(spatial_samples) + 0.5) * dx_mm

    t_open = xs / v_first_mm_s
    t_close = exposure_s + xs / v_second_mm_s

    lit = (
        (t[:, None] >= t_open[None, :])
        & (t[:, None] < t_close[None, :])
    )

    return lit.mean(axis=1, dtype=np.float64)


def make_shot(
    fs: int = 96_000,
    shutter_s: float = 1 / 1000,
    gate_mm: float = 36.0,
    first_travel_ms: float = 4.0,
    second_travel_ms: float | None = None,
    sensor_width_mm: float = 2.0,
    sensor_inset_mm: float = 0.0,
    pre_ms: float = 5.0,
    post_ms: float = 10.0,
    sensor_tau_us: float = 0.0,
    ac_coupling_hz: float = 0.0,
    noise_rms: float = 0.002,
    gain1: float = 0.8,
    gain2: float = 0.8,
    spatial_samples: int = 128,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Return one simulated shutter firing as float32 stereo samples.

    This ndarray can be sent directly into the same analysis code that
    normally receives blocks from sounddevice.
    """
    if second_travel_ms is None:
        second_travel_ms = first_travel_ms

    if fs < 8_000:
        raise ValueError("fs is unrealistically low")
    if gate_mm <= 0:
        raise ValueError("gate_mm must be positive")
    if first_travel_ms <= 0 or second_travel_ms <= 0:
        raise ValueError("curtain travel times must be positive")
    if sensor_width_mm <= 0:
        raise ValueError("sensor_width_mm must be positive")
    if sensor_inset_mm < 0:
        raise ValueError("sensor_inset_mm cannot be negative")
    if sensor_width_mm + sensor_inset_mm > gate_mm / 2:
        raise ValueError("sensor geometry overlaps/exceeds half of the gate")

    v1 = gate_mm / (first_travel_ms / 1000.0)
    v2 = gate_mm / (second_travel_ms / 1000.0)

    # The sensors protrude into the gate from opposite sides.
    sensor1_start = sensor_inset_mm
    sensor1_end = sensor_inset_mm + sensor_width_mm

    sensor2_start = gate_mm - sensor_inset_mm - sensor_width_mm
    sensor2_end = gate_mm - sensor_inset_mm

    # Make the buffer long enough to include both curtains plus padding.
    total_s = (
        pre_ms / 1000.0
        + max(first_travel_ms, second_travel_ms) / 1000.0
        + shutter_s
        + post_ms / 1000.0
    )

    n = int(math.ceil(total_s * fs))
    t = np.arange(n, dtype=np.float64) / fs - pre_ms / 1000.0

    ch1 = sensor_signal(
        t=t,
        x_start_mm=sensor1_start,
        x_end_mm=sensor1_end,
        v_first_mm_s=v1,
        v_second_mm_s=v2,
        exposure_s=shutter_s,
        spatial_samples=spatial_samples,
    )

    ch2 = sensor_signal(
        t=t,
        x_start_mm=sensor2_start,
        x_end_mm=sensor2_end,
        v_first_mm_s=v1,
        v_second_mm_s=v2,
        exposure_s=shutter_s,
        spatial_samples=spatial_samples,
    )

    # Optional response time of the sensor / front-end.
    tau_s = sensor_tau_us * 1e-6
    ch1 = one_pole_lowpass(ch1, fs, tau_s)
    ch2 = one_pole_lowpass(ch2, fs, tau_s)

    ch1 *= gain1
    ch2 *= gain2

    # Optional audio-interface AC coupling.
    if ac_coupling_hz > 0:
        ch1 = one_pole_highpass(ch1, fs, ac_coupling_hz)
        ch2 = one_pole_highpass(ch2, fs, ac_coupling_hz)

    if rng is None:
        rng = np.random.default_rng()

    if noise_rms > 0:
        ch1 += rng.normal(0.0, noise_rms, len(ch1))
        ch2 += rng.normal(0.0, noise_rms, len(ch2))

    stereo = np.column_stack((ch1, ch2)).astype(np.float32)

    # Keep safe headroom for WAV/audio output.
    peak = float(np.max(np.abs(stereo)))
    if peak > 0.98:
        stereo *= 0.98 / peak

    center1 = (sensor1_start + sensor1_end) / 2.0
    center2 = (sensor2_start + sensor2_end) / 2.0

    meta = {
        "fs": fs,
        "shutter_s": shutter_s,
        "gate_mm": gate_mm,
        "v_first_mm_s": v1,
        "v_second_mm_s": v2,
        "sensor1_mm": (sensor1_start, sensor1_end),
        "sensor2_mm": (sensor2_start, sensor2_end),
        "sensor_center_spacing_mm": center2 - center1,
        "ideal_center_delay_s": (center2 - center1) / v1,
        "slit_width_at_nominal_speed_mm": v1 * shutter_s,
    }

    return stereo, meta


def concatenate_shots(
    shots: list[np.ndarray],
    fs: int,
    interval_s: float,
) -> np.ndarray:
    """Place shots into one stereo stream at a fixed start-to-start interval."""
    if not shots:
        return np.zeros((0, 2), dtype=np.float32)

    if interval_s <= 0:
        raise ValueError("interval_s must be > 0")

    step = int(round(interval_s * fs))
    longest = max(len(s) for s in shots)

    if step < longest:
        raise ValueError(
            f"Interval ({interval_s:.3f} s) is shorter than a generated shot "
            f"({longest / fs:.3f} s). Increase --interval."
        )

    total = step * (len(shots) - 1) + len(shots[-1])
    out = np.zeros((total, 2), dtype=np.float32)

    for i, shot in enumerate(shots):
        start = i * step
        out[start:start + len(shot)] += shot

    return out


def write_wav(path: str | Path, samples: np.ndarray, fs: int) -> None:
    """Write stereo 16-bit PCM WAV using only the Python standard library."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate the stereo line-in signal from a two-point shutter tester."
    )

    parser.add_argument(
        "--speeds",
        nargs="+",
        default=["1/60", "1/125", "1/250", "1/500",
                 "1/1000", "1/2000", "1/4000", "1/8000"],
        help="Shutter speeds to generate, e.g. --speeds 1/250 1/1000 1/8000",
    )

    parser.add_argument("--fs", type=int, default=96_000)

    parser.add_argument(
        "--gate-mm",
        type=float,
        default=36.0,
        help="Film-gate dimension ALONG curtain travel. Use 24 for vertical travel, 36 for horizontal.",
    )

    parser.add_argument(
        "--travel-ms",
        type=float,
        default=4.0,
        help="First-curtain traversal time across the complete gate.",
    )

    parser.add_argument(
        "--second-travel-ms",
        type=float,
        default=None,
        help="Second-curtain traversal time. Default: same as first curtain.",
    )

    parser.add_argument("--sensor-width-mm", type=float, default=2.0)
    parser.add_argument("--sensor-inset-mm", type=float, default=0.0)

    parser.add_argument(
        "--sensor-tau-us",
        type=float,
        default=0.0,
        help="Optional first-order sensor/electronics response time.",
    )

    parser.add_argument(
        "--ac-coupling-hz",
        type=float,
        default=0.0,
        help="Optional sound-card high-pass approximation. 0 disables it.",
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=0.002,
        help="Gaussian RMS noise relative to full scale.",
    )

    parser.add_argument("--gain1", type=float, default=0.8)
    parser.add_argument("--gain2", type=float, default=0.8)

    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Start-to-start interval between simulated firings.",
    )

    parser.add_argument("--wav", default="simulated_shutter_sequence.wav")
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument(
        "--play",
        action="store_true",
        help="Play the generated stereo stream through sounddevice.",
    )

    args = parser.parse_args()

    exposure_times = [parse_speed(s) for s in args.speeds]
    rng = np.random.default_rng(args.seed)

    shots = []
    metadata = []

    for speed_text, exposure_s in zip(args.speeds, exposure_times):
        shot, meta = make_shot(
            fs=args.fs,
            shutter_s=exposure_s,
            gate_mm=args.gate_mm,
            first_travel_ms=args.travel_ms,
            second_travel_ms=args.second_travel_ms,
            sensor_width_mm=args.sensor_width_mm,
            sensor_inset_mm=args.sensor_inset_mm,
            sensor_tau_us=args.sensor_tau_us,
            ac_coupling_hz=args.ac_coupling_hz,
            noise_rms=args.noise,
            gain1=args.gain1,
            gain2=args.gain2,
            rng=rng,
        )

        shots.append(shot)
        metadata.append((speed_text, meta))

    stream = concatenate_shots(shots, args.fs, args.interval)
    write_wav(args.wav, stream, args.fs)

    print(f"Wrote: {args.wav}")
    print(f"Sample rate: {args.fs} Hz")
    print()

    for i, (speed_text, meta) in enumerate(metadata, start=1):
        print(
            f"{i:02d}. {speed_text:>7}  "
            f"exposure={meta['shutter_s'] * 1e6:8.1f} us  "
            f"slit={meta['slit_width_at_nominal_speed_mm']:6.3f} mm"
        )

    first_meta = metadata[0][1]
    print()
    print(
        f"First curtain:  {first_meta['v_first_mm_s'] / 1000:.3f} m/s"
    )
    print(
        f"Second curtain: {first_meta['v_second_mm_s'] / 1000:.3f} m/s"
    )
    print(
        f"Sensor 1: {first_meta['sensor1_mm'][0]:.2f}.."
        f"{first_meta['sensor1_mm'][1]:.2f} mm"
    )
    print(
        f"Sensor 2: {first_meta['sensor2_mm'][0]:.2f}.."
        f"{first_meta['sensor2_mm'][1]:.2f} mm"
    )
    print(
        "Sensor center spacing: "
        f"{first_meta['sensor_center_spacing_mm']:.2f} mm"
    )
    print(
        "Ideal first-curtain center-to-center delay: "
        f"{first_meta['ideal_center_delay_s'] * 1e3:.3f} ms"
    )

    if args.play:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise SystemExit(
                "sounddevice is not installed. Run: pip install sounddevice"
            ) from exc

        sd.play(stream, args.fs)
        sd.wait()


if __name__ == "__main__":
    main()
