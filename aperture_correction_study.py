"""Offline finite-aperture study; this does not change live measurement results.

Run: python -B aperture_correction_study.py
Outputs a numerical table and a figure in debug_captures/.
See APERTURE_CORRECTION.md for derivations, assumptions and primary sources.
"""
from pathlib import Path
import json
import math

import numpy as np


def uncovered_fraction(position, profile):
    """Fraction of a uniformly weighted aperture uncovered by a straight edge.

    position=0 is first contact; position=1 is complete clearance.
    """
    u = np.clip(np.asarray(position, dtype=float), 0, 1)
    if profile == "rectangle":
        return u
    if profile == "circle":
        z = 2 * u - 1
        return (np.arcsin(z) + z * np.sqrt(np.maximum(0, 1 - z * z)) + math.pi / 2) / math.pi
    raise ValueError("Unknown aperture profile")


def optical_pulse(times, exposure_s, crossing_open_s, crossing_close_s, profile):
    # Opening and closing times refer to the aperture CENTER, not first contact.
    opened = uncovered_fraction(times / crossing_open_s + 0.5, profile)
    closed = uncovered_fraction((times - exposure_s) / crossing_close_s + 0.5, profile)
    return np.maximum(0, opened - closed)


def half_height_width(times, values):
    peak = int(np.argmax(values))
    threshold = values[peak] / 2
    left = right = peak
    while left > 0 and values[left - 1] >= threshold:
        left -= 1
    while right + 1 < len(values) and values[right + 1] >= threshold:
        right += 1
    return float(times[right] - times[left])


def run_study():
    aperture_mm = 2.0
    velocity_mm_s = 24.0 / 0.0033
    crossing_s = aperture_mm / velocity_mm_s
    dt = 0.00000025
    results = []
    print("Ideal uniform illumination, known local velocity, no sensor/electrical distortion")
    print("profile      true us   full support us   support-corrected us   area us   FWHM us")
    for profile in ("rectangle", "circle"):
        for exposure in (0.001, 0.0005, 0.00025, 0.000125, 0.0000625):
            times = np.arange(-crossing_s, exposure + crossing_s, dt)
            values = optical_pulse(times, exposure, crossing_s, crossing_s, profile)
            lit = np.flatnonzero(values > 1e-9)
            support = float(times[lit[-1]] - times[lit[0]])
            corrected = support - crossing_s
            area = float(np.trapezoid(values, times))
            width = half_height_width(times, values)
            assert abs(corrected - exposure) <= 2.1 * dt
            assert abs(area - exposure) < dt
            result = dict(profile=profile, true_exposure_s=exposure, crossing_s=crossing_s,
                          full_support_s=support, corrected_exposure_s=corrected,
                          normalized_area_s=area, half_height_width_s=width)
            results.append(result)
            print(f"{profile:10s} {exposure*1e6:9.3f} {support*1e6:17.3f} "
                  f"{corrected*1e6:22.3f} {area*1e6:9.3f} {width*1e6:10.3f}")

    # Different curtain speeds: subtract half the SUM of crossing times.
    crossing_open = aperture_mm / 7000
    crossing_close = aperture_mm / 8000
    exposure = 0.000125
    times = np.arange(-0.001, 0.002, dt)
    values = optical_pulse(times, exposure, crossing_open, crossing_close, "circle")
    lit = np.flatnonzero(values > 1e-9)
    corrected = times[lit[-1]] - times[lit[0]] - (crossing_open + crossing_close) / 2
    assert abs(corrected - exposure) <= 2.1 * dt
    assert abs(np.trapezoid(values, times) - exposure) < dt
    print(f"Different speeds, circular aperture: true 125 us, corrected {corrected*1e6:.3f} us")

    # AC coupling removes DC and hence the integral over a complete response.
    # Demonstrate recovery only when the filter and initial dark state are known.
    sample_dt = 0.000001
    ac_times = np.arange(-0.001, 0.101, sample_dt)
    optical = optical_pulse(ac_times, exposure, crossing_s, crossing_s, "circle")
    cutoff_hz = 35.0
    alpha = math.exp(-2 * math.pi * cutoff_hz * sample_dt)
    recorded = np.zeros_like(optical)
    for i in range(1, len(recorded)):
        recorded[i] = alpha * (recorded[i - 1] + optical[i] - optical[i - 1])
    # Exact inverse of the discrete recurrence used immediately above.
    prior_integral = np.concatenate(([0.0], np.cumsum(recorded[:-1])))
    recovered = recorded / alpha + (1 / alpha - 1) * prior_integral
    assert np.max(np.abs(recovered - optical)) < 1e-10
    print(f"AC coupling example: true area {np.trapezoid(optical,ac_times)*1e6:.6f} us; "
          f"raw whole-response area {np.trapezoid(recorded,ac_times)*1e6:.6f} us; "
          f"restored area {np.trapezoid(recovered,ac_times)*1e6:.6f} us")

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    plot_times = np.arange(-0.0003, 0.0006, dt)
    for profile in ("rectangle", "circle"):
        signal = optical_pulse(plot_times, exposure, crossing_s, crossing_s, profile)
        axes[0].plot(plot_times * 1e6, signal, label=f"{profile}, 125 us exposure")
    axes[0].axvspan(0, exposure * 1e6, alpha=0.12, color="green", label="exposure at aperture center")
    axes[0].set_xlabel("Time (microseconds)")
    axes[0].set_ylabel("Fraction of fully illuminated signal")
    axes[0].set_title("125 us exposure through a 275 us aperture response")
    axes[0].legend(fontsize=8)
    for profile, marker in (("rectangle", "o"), ("circle", "s")):
        group = [r for r in results if r["profile"] == profile]
        truth = [r["true_exposure_s"] * 1e6 for r in group]
        axes[1].plot(truth, [r["half_height_width_s"] * 1e6 for r in group], marker=marker,
                     label=f"{profile}: half-height width")
        axes[1].plot(truth, [r["corrected_exposure_s"] * 1e6 for r in group], linestyle="--",
                     label=f"{profile}: support correction")
    axes[1].plot([0, 1000], [0, 1000], color="black", alpha=0.4, label="correct exposure")
    axes[1].set_xlabel("Known exposure (microseconds)")
    axes[1].set_ylabel("Measured estimate (microseconds)")
    axes[1].set_title("Aperture size is correctable under a known response")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    output = Path(__file__).resolve().parent / "debug_captures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "aperture_correction_study.png", dpi=150)
    (output / "aperture_correction_ideal_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("All ideal-model checks passed. These are NOT calibration of the recorded hardware.")


if __name__ == "__main__":
    run_study()
