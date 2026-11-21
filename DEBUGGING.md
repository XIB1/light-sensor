# Capturing test data

AC analysis starts in **AC transient edges** mode. Auto and AC pulse fit remain
selectable. The finite-aperture correction research, its assumptions, and the
SFH309/round-hole considerations are documented in [APERTURE_CORRECTION.md](APERTURE_CORRECTION.md).
The broader source review and proposed circuit validation are in
[SHUTTER_MEASUREMENT_LITERATURE.md](SHUTTER_MEASUREMENT_LITERATURE.md).

## Reference-based light integration

Click **Capture reference from next shot**. The tool selects 1/60 if a faster
speed was selected; set the physical camera to the displayed slow speed and
fire once. A slower selected speed is also supported if its reference passes
the quality checks. In Simulator mode the reference is generated immediately.

After a valid reference, **Light integration (reference)** is selected and the
previous test speed is restored in the tool. Return the camera to that speed
before firing. Ordinary live measurements continue automatically.

Each sensor's exposure is its reconstructed light integral divided by its own
full-open reference level. Curtain travel still uses the detected edge times;
their separation on the graph is not the integrated exposure duration.
AC mode estimates a first-order high-pass decay separately for each channel
from the slow illuminated plateau. DC/direct mode integrates directly after
dark subtraction. No pulse-shape fit or fixed aperture-time subtraction is
used for these exposure results.

Keep illumination, gain, circuit and sensor position fixed. A dimmer unresolved
fast pulse cannot be distinguished from less exposure without an independent
light monitor. In particular, a flat reference and successful reconstruction
do not independently prove phototransistor linearity or hardware accuracy.
Changing source, signal mode, camera or geometry clears the reference. Changes
outside the app require manually capturing a new reference. References last
for the current session.

Weak, digitally clipped or unstable references are rejected, along with
nonphysical or window-sensitive reconstructions. A resolved full-open test
also checks its brightness against the reference. Rejection never silently
falls back to edge exposure. Reference shots do not count toward summary or
speed-button completion. Capturing or clearing a reference preserves the
summary and speed-button completion history, so testing can continue after
repositioning the setup and taking a fresh reference. All measurement methods
and inputs contribute to the same sheet; switching either never hides earlier
tests. Reference shots themselves remain excluded.

Debug reports include the complete numeric `settings.integration_reference`,
its context and identifier. Reference captures have `outcome: "reference"`;
normal integrated results contain `result.integration_diagnostics`, including
integration bounds, area, reference level, AC cutoff and baseline checks.

Enable **Debug mode: save every test** in the Input panel before testing.
Each analyzed simulator or live test creates two files in `debug_captures/`
beside `shutter_tester.py`:

- `.npz`: original stereo samples, the trimmed samples used for analysis,
  and the JSON report embedded as `metadata_json`. Samples keep their recorded
  polarity and floating-point precision.
- `.json`: settings, sample rate, calculated results, event indices, derived
  stop errors and curtain speeds, and the displayed assessment. Rejected
  analyses include the error and their samples too.

Idle listening does not create files. The Input panel reports the last saved
filename or a recording error. Debug mode starts off; existing captures are
kept when it is disabled. Captures are excluded from Git.

Sensor spacing defaults to the confirmed **21 mm center to center** and the
optical aperture defaults to **2 mm**. Each report records the actual settings
used for that test.

`result.event_indices` and `event_times_s` are relative to the trimmed `samples`.
`raw_event_indices` are relative to `raw_samples`; the difference is
`analysis_start_frame`. `captured_at_utc` is the recording timestamp, not the
time of the opening edge.

The UI orders the sensor boxes and waveform labels by detected opening time:
the sensor reached first appears on the left, followed by the average and the
sensor reached second on the right. Each box and waveform identifies its
physical input channel. Equal opening times are labelled as unresolved order.
Each exposure box judges its own reading against the selected stop tolerance.
The overall exposure verdict requires both sensors and the average to pass;
a passing average cannot hide a failing sensor. Changing exposure tolerance
reassesses the displayed capture using its original marked speed.
Raw sample columns, result fields such as `exposure_sensor_1_s`, and event keys
such as `sensor1_open` retain physical input-channel numbering. New reports
include `sensor_order` to map the displayed first/second sensors back to those
channels; older captures can infer that mapping from their opening indices.

To replay a completed capture:

```python
import json
import numpy as np
from shutter_tester import analyze_signal

with np.load("debug_captures/REPLACE_WITH_CAPTURE.npz", allow_pickle=False) as capture:
    report = json.loads(str(capture["metadata_json"]))
    s = report["settings"]
    result = analyze_signal(
        capture["samples"], s["nominal_s"], s["expected_travel_ms"],
        s["reference_distance_mm"], s["sensor_spacing_mm"],
        s["optical_aperture_mm"], s["coupling"],
        fs=s["sample_rate_hz"], ac_method=s["ac_method"],
        integration_reference=s.get("integration_reference"),
    )
print(result)
```

Summary min, average, and max cells show milliseconds and signed stops relative
to the marked speed. Positive means longer exposure; negative means shorter.
The average stop error is calculated from the average exposure duration.
Only speeds with completed measurements are listed, in order from slowest to
fastest. The selected latest-test count still applies across reference refreshes;
use **Clear summary** to start a fresh sheet. Camera, geometry, method, input,
reference and tolerance changes never clear or switch the sheet.

The latest 20 completed tests and their recorded pass/fail results per speed,
plus the selected display count, are saved automatically to
`shutter_summary.json` beside the script and restored after restarting the app.
**Clear summary** also saves the empty sheet, so cleared results do not return
on the next launch. The file is excluded from Git. A saving error is shown on
the summary page while results remain in memory. An unreadable existing file
is preserved and is not overwritten by new measurements; **Clear summary**
explicitly permits starting over. Versions before this persistence change kept
results only in memory and cannot automatically restore unsaved earlier sheets.

Transient-edge measurements mark the local midpoint of each opening/closing
transition, rather than a peak that may occur later on a flat or decaying tail.
For pulse fitting with separated transitions, the AC decay is estimated from
each channel's illuminated plateau. `result.fitted_ac_cutoff_hz` records those
two estimates. Overlapping short pulses retain the default response assumption
because there is no resolved plateau to estimate from. Pulse-fit markers are
model estimates; transition markers are midpoint measurements.

Shutter-speed buttons display a checkmark when the latest selected number of
completed measurements on the current sheet all recorded overall PASS.
There must be a full window (for example, five passing measurements for 5/5).
A measured FAIL removes the checkmark until it falls outside the window.
Changing the summary count updates checkmarks immediately; changing input or
measurement method keeps them. Clear summary clears them too. These are
recorded verdicts under the settings used for each test, not retroactive
assessments under newly chosen tolerances.
Rejected analyses do not enter the completed-measurement summary.
