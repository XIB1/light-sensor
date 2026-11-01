# Capturing test data

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
    )
print(result)
```

Summary average, min, and max cells show milliseconds and signed stops relative
to the marked speed. Positive means longer exposure; negative means shorter.
The average stop error is calculated from the average exposure duration.
