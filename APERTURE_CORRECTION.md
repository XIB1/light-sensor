# Recovering exposure through a finite sensor aperture

Research for the current rig: two SFH309 phototransistors, each covered by
heat-shrink tubing leaving a round 2 mm entrance opening, spaced 21 mm C-C.

**Broader literature review:** see
[SHUTTER_MEASUREMENT_LITERATURE.md](SHUTTER_MEASUREMENT_LITERATURE.md).
The revised recommendation is to prioritize integrated light divided by a
stable full-open reference, after characterizing the actual circuit and AC
path. Published implementations explicitly handle overlapping fast pulses
this way. The earlier waveform fits below remain exploratory.

The supplied layout specifies 3 V, collector load resistors, 1-10 microfarad
coupling capacitors and 1 kilohm series resistors. The user now believes the
collector resistors were 1 kilohm rather than the layout's 10 kilohms; actual
values remain unconfirmed. At 3 V, 1 kilohm gives approximately 3 mA of static
current headroom rather than 0.3 mA for 10 kilohms. The complete assembled
response still needs characterization.

**Live implementation:** reference-based light integration is now available.
Use **Capture reference from next shot**, set the camera to the displayed
1/60-or-slower speed, and fire once. A valid reference selects integration;
then return the camera to the restored test speed. See
[the operating and replay instructions](DEBUGGING.md#reference-based-light-integration).
AC reconstruction uses an estimated first-order filter, with quality checks;
this is a reference-based estimate, not independent hardware calibration.

**A known aperture can be compensated. A 2 mm opening is not a fundamental
minimum measurable exposure.** The earlier statements treating its crossing
time as a hard exposure limit were too restrictive. What fails near that
duration is an uncompensated midpoint/pulse-width interpretation, not every
possible measurement of the complete waveform.

Transient edges is now the default in both the application and `analyze_signal`.
This change does not silently apply a new correction to existing measurements.
The earlier experiments below are offline mathematical validation, not a
calibration of the physical hardware or certification of previous fast captures.
Their empirical pulse fits are distinct from the new integrated-light mode.

## 1. Which time is being measured?

There are three different quantities:

1. Exposure at the center of an aperture: the interval between a curtain
   uncovering that point and the other covering it.
2. First-light to last-light duration over the whole aperture: longer because
   a moving slit can illuminate one side before reaching the other.
3. Effective exposure: integrated light divided by the fully open light level.
   Across a finite detector, this is a sensitivity-weighted spatial average.

ISO 516 distinguishes effective exposure from total illumination duration and
defines effective time using exposure divided by unobstructed illuminance.
That distinction supports integration; it does not say to subtract detector
diameter from an arbitrary threshold duration. See the [original ISO preview,
sections 3.3–3.5](https://cdn.standards.iteh.ai/samples/70966/498843d5579f43708e741c0ed5f248e8/ISO-516-2019.pdf).
Its separate lens/light-cone terms are not our 2 mm detector opening.

## 2. The geometric correction

The following is a derivation for a straight translating shutter slit and a
detector aperture. It assumes known local curtain velocities, negligible
unaccounted electrical broadening, and detectable illumination over the
full stated aperture width. Width means extent **along curtain travel**.

For equal curtain velocities:

```
a = aperture width along travel, mm
v = local curtain velocity, mm/s
C = a / v                  aperture crossing time
W = first-light-to-last-light duration over the aperture
T = point exposure at the aperture center

W = T + C
T = W - a / v
```

For different opening and closing velocities:

```
T = W - (a / 2) * (1 / v_open + 1 / v_close)
```

For a centered aperture the opening's first contact occurs `a/(2*v_open)`
before its center is uncovered, and closing's last contact occurs
`a/(2*v_close)` after its center is covered. Subtracting those two margins
gives the expression above.

Our measured C-C delays can supply a first velocity estimate:

```
d = sensor center spacing
D_open, D_close = corresponding intersensor time delays

v_open  = d / abs(D_open)
v_close = d / abs(D_close)

T = W - a / (2*d) * (abs(D_open) + abs(D_close))
```

This uses the physical 21 mm spacing directly. No full-gate extrapolation is
necessary. However, the delays give an **average velocity between sensors**;
using it for both local aperture crossings is an additional assumption if
curtain velocity changes with position.

Example: 24 mm in 3.3 ms corresponds to 7272.7 mm/s. A 2 mm aperture then
has `C = 0.275 ms`. A true `0.125 ms` exposure produces an ideal optical
support lasting `0.400 ms`; subtracting the known crossing time recovers
`0.125 ms`. Exposure shorter than the aperture crossing is recoverable in
this ideal case.

## 3. Why not subtract 0.275 ms from the current reading?

The current edge detector measures transition **midpoints**, not first and
last optical contact. They are different widths.

For a uniformly weighted rectangular aperture, with equal velocities:

```
T >= C:  half-height width = T
T <  C:  half-height width = C
```

Subtracting `C` from either case is not the geometric support correction.
Once opening and closing overlap, our local midpoint implementation can also
be influenced by the neighboring transition. That explains the earlier
synthetic 250 us -> 302 us result; it does not mean an aperture correction is
impossible.

## 4. The round hole needs its own response

For an ideal circular aperture under uniform illumination and sensitivity,
the projected sensitivity along travel is proportional to a circle's chord
length. A moving edge therefore uncovers a circular segment rather than a
constant-width strip.

If `u` goes from 0 at first contact to 1 at full clearance, let `z = 2*u - 1`.
The ideal uncovered fraction is

```
F(u) = [asin(z) + z*sqrt(1-z*z) + pi/2] / pi,   0 <= u <= 1
```

The optical signal from the moving slit is the difference between the opening
and closing fractions. With opening center time 0 and closing center time T:

```
s(t) = F(t/C_open + 1/2) - F((t-T)/C_close + 1/2)
```

This expression requires a nonnegative optical opening over the aperture;
severe capping needs explicit occlusion handling. The full-support correction
still applies when those boundaries are observable, but the circular
half-height widths differ from rectangular ones. The live template presently
uses uniform samples across a line segment, so it does not implement this
circular area response.

## 5. Integration avoids the width ambiguity

Let `s(t)` be the baseline-corrected optical signal divided by the signal that
the same detector would produce when fully illuminated under the same light.
Then

```
T_effective = integral s(t) dt
```

For spatially uniform exposure, integrating each point's contribution first
shows that this integral equals T regardless of aperture shape. A larger
aperture spreads light over time but does not change the normalized total.
This is why normalizing by a **fully open reference** matters. Normalizing by
the peak of a short pulse is not equivalent: a narrow slit may never uncover
the entire aperture.

For spatially varying exposure the integral gives a weighted average across
the aperture, not automatically the exposure at one point. A symmetric aperture
with symmetric weighting averages a linear exposure gradient to its center
value; nonlinear clipping/capping or asymmetric illumination needs more care.

A linear low-pass sensor response with known constant DC gain preserves this
area if the full response is captured. Saturation, automatic gain changes and
changing illumination invalidate that simple normalization.

Our sound input is AC coupled. Its complete pulse response has approximately
zero net area, so we cannot integrate the signed audio recording directly.
For a first-order high-pass response with cutoff f:

```
x(t) = y(t) + 2*pi*f * integral_from_dark_baseline_to_t y(u) du
```

Here y is the recorded AC signal and x is the reconstructed pre-AC signal,
assuming the initial state is known. Real hardware needs measured filter
behavior and careful baseline handling; integration amplifies baseline drift.
Our resolved captures indicate cutoffs near 2–3 Hz rather than the original
35 Hz assumption. The existing light-area comparisons remain relative
diagnostics because illumination levels changed between shots.

## 6. What the SFH309 changes

The [OSRAM SFH309 datasheet, page 3](https://www.tme.eu/Document/419f75daadd74ab5572d3cb11e18162f/SFH%20309_EN.pdf)
lists a 0.038 mm² sensitive area and typical 7 us rise/fall times measured at
1 mA, 5 V and a 1 kΩ load. Its package and directional characteristics are
also documented there. Those values do not establish the response of an
unknown assembled circuit behind a heat-shrink entrance mask.

My inference: a round 2 mm entrance window does not by itself prove uniform
sensitivity over a 2 mm disk in the shutter plane. The package optics,
recess, ray angles and illumination determine the effective weighting.
The much smaller chip area must **not** simply replace 2 mm in the settings:
it is an internal sensor area, not a measured projected footprint.

Also, our `SENSOR_TAU_US = 7` is a one-pole time constant. A one-pole 10–90%
rise time is `ln(9)*tau`, so it is not the same as the datasheet's 7 us rise
time. This is an existing modeling assumption, not an electrical calibration;
changing it to a datasheet-derived value alone would not validate the circuit.

## 7. Numerical verification

Run `python -B aperture_correction_study.py`. This independent analytic study
uses both uniform rectangles and circles, including exposures shorter than
the crossing time. With a 275 us crossing response and 0.25 us simulation
sampling, representative results are:

| Aperture | Known exposure | Total optical support | Support minus crossing | Integrated normalized light |
|---|---:|---:|---:|---:|
| Rectangle | 500 us | 774.5 us | 499.5 us | 500.0 us |
| Rectangle | 250 us | 524.5 us | 249.5 us | 250.0 us |
| Rectangle | 125 us | 399.5 us | 124.5 us | 125.0 us |
| Circle | 500 us | 774.5 us | 499.5 us | 500.0 us |
| Circle | 250 us | 524.5 us | 249.5 us | 250.0 us |
| Circle | 125 us | 399.5 us | 124.5 us | 125.0 us |

The sub-microsecond support error comes from excluding the boundary samples.
These are ideal-model numerical errors, not expected errors of the 96 kHz
hardware. The script also verifies unequal curtain velocities and exact
reconstruction through a known discrete AC filter. Its figure is saved as
`debug_captures/aperture_correction_study.png`.

## 8. Implementation direction after the broader literature review

The [broader review](SHUTTER_MEASUREMENT_LITERATURE.md) changes the priority:
make normalized light area the proposed main exposure measurement, with
waveform fitting as a separately validated diagnostic.

1. Characterize each assembled channel with independently known optical
   pulses and stable illumination. Record the actual circuit values and check
   linearity, saturation headroom and brightness-dependent response.
2. Obtain a valid full-open light reference for each channel. For fast pulses
   that never fully illuminate the aperture, retain the reference from a slower
   exposure under the same conditions, rather than use the fast pulse's peak.
3. Measure effective exposure by integrating the reconstructed light signal
   divided by that reference. Use a DC-capable receiver or independently
   characterize and validate the complete AC reconstruction first.
4. Retain curtain timing as a separate quantity using the 21 mm intersensor
   delays. For point exposure or waveform modeling, characterize spatial
   weighting and local velocity as well as electrical response.
5. Validate against an independent method and include reference, baseline,
   response and sampling uncertainty. Reject inconsistent fits instead of
   turning them into an authoritative corrected PASS/FAIL result.

This is feasible with a 2 mm hole in principle. Finite noise, bandwidth,
sampling and unknown response still impose an accuracy limit; knowing the
hole diameter does not remove those uncertainties. Exposure and gain become
hard to distinguish for pulses much shorter than a poorly known response.

## 9. Check against the new real captures

The ten shots at 1/4000, 1/2000 and 1/1000 recorded on 2026-11-01 at
17:14 UTC provide a useful check beyond the ideal simulation. Full results,
reproduction scripts and waveform comparisons are in
[the real-capture validation report](debug_captures/latest_aperture_validation.md).
That report and its raw captures are local debug artifacts ignored by Git.

All stored event timing, averages, stop errors, mismatch and 21 mm C-C travel
calculations replay and recompute correctly. However, the resolved 1/1000
opening transitions take about 197 us at the first location and 98 us at the
second; closing transitions take 222 and 117 us. A uniform round aperture
using the intersensor mean velocity would predict about 173 us for each
10-90% optical transition. Thus one shared geometric response does not
describe this assembled rig. Local velocity, optical weighting and circuit
response remain separate from the validated physical dimensions.

An offline experiment extracts separate opening/closing step profiles for each
channel from the resolved 1/1000 shots, then fits their difference to each fast
waveform with free amplitude and timing. This produces close waveform agreement
without using the fast shot to construct its own profile. Mean results:

| Selected speed | Current edge time, first / second | Empirical-response fit, first / second |
|---|---:|---:|
| 1/1000 | 867 / 833 us | 870 / 835 us |
| 1/2000 | 375 / 330 us | 371 / 331 us |
| 1/4000 | 243 / 153 us | 129 / 85 us |

The last column is **exploratory, not calibrated exposure**. In particular,
using different individual reference shots moves the 1/4000 estimates across
110-158 us at the first sensor and 58-102 us at the second. The estimates also
assume response shape transfers between shots and that the step midpoints
represent the intended optical center convention. This experiment supports
response-aware correction as an approach; it does not certify the current
round-hole idealization or a shutter-capping diagnosis.

The practical conclusion is that transient edges agrees with the empirical
response analysis at 1/1000 and 1/2000 in this batch, but at 1/4000 a displayed
243 us should not be interpreted as proof of a correct 250 us exposure.
At the time of those capture audits, no correction was applied to live results.
The new live integration mode requires a newly captured, valid reference; it
does not retroactively change those archived measurements.

### Follow-up batch and response stability

Thirteen further captures at 19:32-19:34 Europe/Helsinki on 2026-11-01
(17:32-17:34 UTC in archive filenames) replay exactly with transient edges.
See the [follow-up validation report](debug_captures/repeat_aperture_validation.md).

| Selected speed | Shots | Current edges, first / second |
|---|---:|---:|
| 1/1000 | 4 | 872 / 898 us |
| 1/2000 | 4 | 375 / 378 us |
| 1/4000 | 5 | 192 / 160 us |

The user made no intentional changes, but the setup may have moved slightly.
The first channel's mean 1/1000 peak is about eight times lower, its opening
transition changes from 197 to 108 us, and its closing transition from 222 to
277 us. A scalar gain adjustment alone does not describe those shape changes.
Movement affecting illumination or optical weighting is plausible, but these
captures do not establish the cause.

Response fits using the new batch's own 1/1000 profiles remain close to edges
at 1/1000 and 1/2000. At 1/4000, however, estimates change substantially with
the chosen batch's reference, and some same-batch fits predict unphysical
negative restored light. They must not be promoted to calibrated exposure
readings. This is a concrete limit on transferring the earlier experimental
fit to new captures, not a failure of the ideal geometric derivation.

The apparent intersensor exposure difference also changes sign at 1/1000
and nearly disappears at 1/2000. This further weakens a consistent capping
diagnosis from the current widths alone. Stable positioning and illumination,
independent response validation, and rejection of inconsistent fast fits are
needed before applying a correction in the live app.

Two additional 1/4000 shots at 19:42 Europe/Helsinki give recent-reference
model outputs of 124 / 140 us and 160 / 158 us, versus displayed edge widths
of 188 / 177 us and 219 / 177 us. The last shot is encouraging as an
intersensor consistency check. However, the first shot's CH1 fit predicts
negative restored light, and CH1 estimates change substantially with reference
batch. These remain diagnostic outputs, not accepted corrected exposures.
See the [two-shot correction report](debug_captures/latest_two_corrected_fast.md)
and its waveform comparison for the fit quality and reference sensitivity.

## Sources and what they support

- [ISO 516:2019 public preview](https://cdn.standards.iteh.ai/samples/70966/498843d5579f43708e741c0ed5f248e8/ISO-516-2019.pdf),
  sections 3.3–3.5: effective versus total exposure definitions. The detector
  aperture equations above are my derivation, not a quotation of ISO's
  lens/light-cone formula.
- [OSRAM SFH309 manufacturer page](https://ams-osram.com/products/photodetectors/phototransistors/osram-radial-t1-sfh-309)
  and the linked original datasheet mirror above: device characteristics and
  specified test conditions.
- [Simon, Vakulya and Rátosi, Sensors 22(5), 1871 (2022)](https://doi.org/10.3390/s22051871):
  published direct optical measurement experiments, integration methods and
  measurement uncertainty. The article does not certify this tester.
- [Reveni Labs tester manual](https://www.reveni-labs.com/reveni-labs-camera-tester-user-manual),
  Curtain Sensors and Calibrating Curtain Detectors: a manufacturer's practical
  distinction between timed and analog measurements, plus channel balance and
  timing-offset calibration. Their implementation is not our calibration.
