# Measuring fast shutters through a finite optical sensor

Literature review, 2026-11-01. Application: two SFH309 phototransistors,
21 mm C-C, round 2 mm heat-shrink openings, stereo PC line input at 96 kHz.
This review distinguishes published methods, manufacturers' implementation
descriptions, our mathematical deductions, and measurements from this project.

The circuit supplied during this review uses a nominal 3 V battery supply,
a 10 kilohm collector resistor per sensor, emitter to ground, a 1-10 microfarad
coupling capacitor, and a 1 kilohm series resistor into each line-input channel.
The user subsequently clarified that the collector resistors were likely
1 kilohm, and may have fitted different resistors to adjust signal strength;
actual per-channel resistor and capacitor values remain unconfirmed.

## Main finding

The clearest routes supported by the sources are **normalized integrated
light** and **a small, calibrated optical sampling spot**. Full-waveform
correction is possible, but requires an independently characterized response
and an uncertainty analysis. The literature does not establish the experimental
124-160 us estimates from this rig as calibrated shutter times.

For this project I would prioritize measuring effective exposure from light
area and a stable full-open reference. The immediate prerequisite is to
characterize the actual phototransistor/load/audio circuit. The known 2 mm
opening and 21 mm spacing are useful geometry, but do not identify that circuit's
dynamic response or the optical sensitivity distribution.

| Approach | How it addresses broad or overlapping pulses | Requirement for this rig |
|---|---|---|
| Integrated light / full-open level | Measures received exposure without assigning two arbitrary threshold crossings | Linear response, trustworthy reference level, and DC signal or a characterized AC inverse |
| Small optical spot and calibrated timing | Makes geometric transition duration small compared with exposure | Known footprint at the shutter plane and characterized detector/electronics delay |
| Calibrated threshold / constant-fraction timing | Reduces amplitude-dependent timing errors for suitable, stable shapes | Resolved transitions or a validated relationship between threshold width and exposure |
| Calibrated waveform model | Accounts explicitly for optical averaging and electrical filtering | Independent temporal/spatial calibration, physical constraints, and uncertainty propagation |
| Independent photometric or time-encoded tests | Checks the answer through a different observable | A separate suitable reference method; not all digital-camera techniques transfer to a film camera |

These are alternative measurement designs. A reference-based integration mode
was subsequently implemented, with first-order AC reconstruction and quality
checks; independent hardware validation remains necessary. Its operating
instructions are in [DEBUGGING.md](DEBUGGING.md#reference-based-light-integration).

## 1. Define the quantity before selecting an algorithm

[ISO 516:2019, public preview, sections 3.3-3.5](https://cdn.standards.iteh.ai/samples/70966/498843d5579f43708e741c0ed5f248e8/ISO-516-2019.pdf)
distinguishes effective exposure, central exposure time, and total illumination
duration. Effective time is exposure divided by unobstructed illuminance,
`t_effective = H / E0`. The [official ISO catalogue](https://www.iso.org/standard/70966.html)
identifies the 2019 edition as current, confirmed in 2024. The public preview
does not include all the test procedures, so this review is not a complete
audit of compliance with the current standard.

For our detector, three times must remain distinct:

- The interval between the two curtains crossing a particular geometric point.
- The duration of any light over the whole sensed footprint.
- The received light integral divided by that detector's full-open light level.

A finite detector returns a spatially weighted measurement. Its effective time
is not automatically the point exposure at the center of the hole. For uniform
exposure across the footprint the two agree. A symmetric weighting also averages
a linear exposure gradient to its center value; clipping or asymmetric weighting
needs further treatment. These are deductions from the measurement equation,
not an ISO formula for our heat-shrink opening.

ISO's separate taking-lens/curtain-distance terms must not be interpreted as
the diameter of our sensor opening.

## 2. Integration has a direct precedent for this exact symptom

The most directly relevant implementation is the [Baby Shutter Tester MkII
manual, sections 6.1.4 and 6.2.2](https://photographyelectronics.com/wp-content/uploads/2025/08/BabyShutterTester_mkII_UserManual_en_1.0.0_-B.pdf).
It explicitly describes focal-plane pulses that start falling before reaching
the full illumination level. Its solution is to retain `E0` from a slower
exposure and use integrated light divided by that reference. Its global mode
requires unchanged illumination and aperture between measurements. This is a
manufacturer's implementation description, not independent certification of
its accuracy or our hardware.

The designer's [methods article](https://photographyelectronics.com/fr/methods-for-measuring-shutter-speed/)
also describes the requirements for an integration receiver: linear detection
and amplification, sufficient bandwidth and dynamic range, stable illumination,
and operation without saturation. Its gain-cancellation argument does not remove
those conditions.

### Why this handles aperture broadening

The following is our derivation for a stable, linear system. Let `w(x)` be
normalized spatial sensitivity, including illumination weighting, and let
`T(x)` be exposure at each point. Then

```
normalized_light(t) = integral w(x) * shutter_open(x,t) dx
T_effective = integral normalized_light(t) dt
            = integral w(x) * T(x) dx
```

The order of integration can be exchanged. If `T(x)` is constant, the result is
that exposure regardless of hole shape. The hole broadens the waveform without
changing normalized total light. This is why a 2 mm hole is not a fundamental
minimum measurable exposure.

A linear low-pass response with unit DC gain also preserves the complete area:
`integral(h * light) = integral(h) * integral(light)`. This benefit does not apply
automatically to a saturating, nonlinear, history-dependent receiver.

For an actual voltage recording, the proposed measurement is

```
T_effective = integral(V_light(t) - V_dark) dt
              / (V_fully_open - V_dark)
```

Use consistent polarity, gain and illumination. At high speeds, do not replace
the denominator with that short pulse's own peak: the slit may never illuminate
the entire sensed area. Nor can the light integral be inferred from a displayed
midpoint width alone.

## 3. The AC-coupled audio path requires special treatment

The ideal area argument above applies before the AC high-pass response. For a
complete isolated event, a stable high-pass filter removes the DC area; summing
the entire signed audio pulse therefore does not recover optical exposure.
Integrating only whichever lobe is convenient is not the same measurement.

For a known first-order high-pass filter and a known initially dark state:

```
x(t) = y(t) + omega_c * integral_from_initial_time_to_t y(u) du
```

Here `x` is the pre-high-pass signal and `y` the measured AC signal. The equation
follows directly from `H(s) = s / (s + omega_c)`. An offset error in `y` grows
into drift through the integral. Unknown initial state, multiple filters, gain
changes, clipping and nonlinear detection invalidate a naive inverse.

Our resolved recordings suggested an effective decay near 2-3 Hz. That is useful
evidence, but not a full calibration of the external coupling capacitor plus
the PC input. The previously assumed 35 Hz fast template and the approximate
2.5 Hz offline inverse are not independently measured transfer functions.

For integrating light I would prefer a DC-capable linear receiver. The current
audio path remains a possible route if its response and reconstruction are
validated against known optical pulses first. This is an engineering
recommendation for this rig, not a requirement that the literature imposes on
every optical shutter tester.

## 4. A small optical spot is the established alternative

The [Tsubosaka 7FR-80D manufacturer's brochure, page 2](https://www.tsubosaka.co.jp/english/wp-content/uploads/sites/2/2019/06/7FR-80D_eng.pdf)
specifies a 0.1 mm measuring spot at three locations, 7.5 mm apart, with
per-point calibration. It advertises operation to 1/8000 s and measurement
accuracy below +/-5 us under its stated geometry. This shows a commercial
design that controls spatial sampling and calibration; those specifications
are not performance claims for our setup.

Scale calculation using our dimensions and an illustrative measured 2.6 ms
delay over 21 mm:

```
mean velocity = 21 mm / 2.6 ms = 8.08 mm/ms
2.0 mm crossing estimate = 248 us
0.1 mm crossing estimate = 12.4 us
nominal 1/4000 exposure = 250 us
```

Reducing the illuminated extent along travel makes the optical transition
smaller relative to exposure. The relevant footprint is at the shutter plane,
not simply the plastic opening or the internal silicon chip size. A small spot
also does not eliminate electrical delay, saturation, alignment or SNR errors.
These calculations use mean velocity only; they do not establish local velocity
at either detector.

## 5. The circuit explains why resistor values matter

The [OSRAM SFH309 datasheet, page 3](https://www.tme.eu/Document/419f75daadd74ab5572d3cb11e18162f/SFH%20309_EN.pdf)
gives typical 7 us rise and fall times at collector current 1 mA, supply 5 V,
and load 1 kilohm. It also documents the device's sensitive area and directional
response. Those are specified test conditions, not a response calibration for
the battery/load/sound-card assembly.

[Vishay's Application of Optical Sensors, page 6 of the PDF](https://www.newark.com/pdfs/techarticles/vishay/ApplicationNote-ReflectiveSensorApplicationNote.pdf)
explains that phototransistor speed depends on load resistance and junction
capacitance. Saturation can extend turn-off through stored charge; reducing
load resistance increases speed at the expense of voltage signal. This is
manufacturer guidance about the device class, not proof that our SFH309s are
saturating.

### Calculation for the circuit supplied by the user

With a nominal 3 V supply and 10 kilohm collector load, the steady current
available as the collector approaches ground is approximately

```
I_headroom ~ 3 V / 10,000 ohm = 0.3 mA
```

The actual active-region headroom is lower if a collector-emitter voltage margin
is retained. The load, supply and operating current differ substantially from
the datasheet timing test. Different collector resistors fitted to change signal
strength can change timing as well as amplitude. They cannot be modeled merely
as multiplying the saved samples by a scalar.

The user now believes 1 kilohm collector resistors were used. With that value,
the corresponding calculation is `3 V / 1000 ohm = 3 mA`, so the 0.3 mA figure
does not describe that likely configuration. Lower load resistance generally
improves headroom and speed; the actual bias and response still require a
measurement and must not be inferred solely from the parts' nominal values.

This makes load-dependent response and saturation concrete hypotheses to test
alongside optical weighting or movement. The audio waveform not reaching digital
full scale does not prove the upstream transistor stayed linear.

The coupling capacitor is another unknown. In an illustrative linear small-signal
equivalent circuit, with transistor output resistance large compared with the
collector load:

```
f_external_HP ~ 1 / [2*pi*C*(R_collector + R_series + R_line_input)]
```

For example, 1 microfarad with an illustrative total resistance of 58 kilohms
gives 2.74 Hz; 10 microfarads gives 0.274 Hz. The 58 kilohm value is an example,
not a measurement of this PC. The sound card may contain additional coupling
and filtering, and the transistor's effective source impedance need not remain
constant outside its linear operating region.

The existing simulator's 7 us one-pole time constant also must not be equated
with a 7 us 10-90% rise time: for a single pole the latter is `ln(9)*tau`.

[Hamamatsu's Si photodiodes technical note](https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/ssd/si_pd_kspd9001e.pdf),
sections 2-2, 2-6 and 3-2, describes how load, bias and capacitance affect linearity
and speed, and supplies both current-to-voltage and light-integration circuits.
It is a useful reference if a PIN-photodiode receiver is chosen. Changing sensor
type alone is not a complete receiver design or a calibration.

## 6. Timing thresholds are conditional solutions

The [1986 ISO 516 public preview, section 6](https://cdn.standards.iteh.ai/samples/4577/d563f7bd379b49f1af314c88d19d656e/ISO-516-1986.pdf)
requires consistency of the illuminance curve before routine digital timing.
Its front-shutter procedure obtains the threshold corresponding to effective
time from the measured curve; it notes the half-height case for a trapezoid.
The apparatus clauses specify DC response, linearity and substantial bandwidth,
and at least 100 timebase samples during the minimum effective interval.
These are historical clauses; they have not been verified here as unchanged
requirements of the 2019 edition, nor is the front-shutter procedure a universal
formula for our overlapping focal-plane signals.

In a related field, [ORTEC's fast-timing introduction, pages 3-5](https://www.ortec-online.com/-/media/ametekortec/other/fast-timing-discriminator-introduction.pdf)
describes amplitude-dependent threshold timing and constant-fraction
discrimination. Constant-fraction timing suppresses amplitude walk for signals
with consistent rise times. It does not promise immunity to changing shapes.
Its nuclear-detector application is not a ready-made finite-aperture shutter
correction.

For our algorithm, a fractional transition midpoint is a reasonable diagnostic
when the two steps are resolved. A fraction of an overlapped short pulse's own
peak is not a universal center-exposure measurement. Changing the threshold
cannot replace knowledge of the response.

An independent mathematical check illustrates the limitation: for a Gaussian
pulse of peak A and width parameter sigma, area divided by A is
`sqrt(2*pi)*sigma`, while the half-height width is `2*sqrt(2*ln(2))*sigma`.
The latter is about 6.1% shorter. Symmetry alone therefore does not make an
arbitrary pulse's half-height width equal to its normalized light area. This
example does not explain all our observed errors; it shows why threshold
rules must be tied to the actual response and reference definition.

## 7. Waveform inversion is established metrology, with uncertainty

[Eichstaedt et al., Metrologia (2016)](https://www.nist.gov/publications/challenges-uncertainty-evaluation-time-dependent-measurements)
treat dynamic measurements modeled by convolution. They explain why recovering
the input requires regularization in noisy inverse problems and why uncertainty
from that regularization matters. Their examples are force, oscilloscope and
hydrophone measurements, not camera shutters.

[Eichstaedt and Wilkens, JASA 141, 4155-4167 (2017)](https://publica.ptb.de/resources/show/38525)
develop an uncertainty contribution for regularized deconvolution using bounded
prior spectral information. This is a method for accounting for inversion
assumptions, not evidence that our current reference profiles are correct.

[Hale et al., Complete Waveform Characterization at NIST](https://www.nist.gov/system/files/documents/2017/05/09/04574962.pdf)
describe measured response correction, regularized inversion and covariance-based
uncertainty in calibrated waveforms. The transferable lesson is to calibrate the
measurement response and propagate its uncertainty. Their high-speed accuracy
does not transfer to our audio recording.

### A physically constrained forward model for this project

Under a local straight-edge model, a suitable starting point is

```
optical(t) = integral w(x) * 1{t_open(x) <= t < t_close(x)} dx
recorded(t) = baseline + gain * (h_audio * h_sensor * optical)(t) + noise
```

An interval with closing before opening contributes no illumination. The
occlusion must be applied at the spatial point before spatial integration and
electrical filtering. Clipping an already filtered empirical waveform to zero
does not implement that model.

The model needs independently tested temporal response, spatial weighting,
gain linearity and baseline behavior. It should explore uncertainty in those
quantities and in velocity. The 21 mm C-C delay constrains average velocity;
it does not uniquely supply both local aperture-crossing velocities.

Nonnegativity applies to modeled optical light, not to the raw AC voltage:
negative opening transients in these recordings are expected. The negative-light
warnings in our offline fits refer to predicted reconstructed optical light.

Our fits currently learn profiles from other shutter shots without an
independently known optical exposure or receiver response. High waveform
similarity therefore cannot by itself establish absolute timing accuracy.

## 8. Published calibration practices relevant to movement and channels

[Reveni Labs' tester manual, Sensor Heads and Calibrating Curtain Detectors](https://www.reveni-labs.com/reveni-labs-camera-tester-user-manual)
distinguishes digital curtain detectors from its analog sensor and documents
separate balance and offset adjustments. Its balance procedure flips the
sensor 180 degrees while keeping the camera and lighting stable to distinguish
sensor-associated from camera-associated differences. It also describes
offset checking against a trusted camera. This is a maker's practical
calibration procedure, not an independent guarantee that a nominally good
camera is an exact timing reference.

[Fowler and Dezsi, J. Res. NIST 100, 277-283 (1995)](https://nvlpubs.nist.gov/nistpubs/jres/100/3/j13fow.pdf)
use background subtraction and a separate monitor detector to normalize source
drift in radiometric aperture measurements. This is not shutter timing, but it
provides a concrete example of measuring source variation instead of assuming
it absent. A monitor would not correct a sensor that physically moves within
an uneven illumination field.

[Doensberg, Maentynen and Ikonen, Optical Review 23, 510-521 (2016)](https://link.springer.com/article/10.1007/s10043-016-0181-2)
describe optical aperture determination in the assembled measurement geometry
and discuss differences between geometric and effective aperture area from
edge shape and light collection. This supports characterizing the assembled
optical response. It does not establish the magnitude or cause of any such
effect in our heat-shrink masks.

The physical 2 mm diameter remains accepted. The implication is to measure
how the assembly weights illumination under the actual geometry, not to
substitute the SFH309 chip diameter or assume the hole measurement was wrong.

## 9. Independent methods and spatial exposure maps

[Simon, Vakulya and Ratosi, Sensors 22(5), 1871 (2022)](https://doi.org/10.3390/s22051871)
review and experimentally compare direct optical timing, moving targets,
time-encoded LED patterns and equivalent sampling. Their direct-method
analysis relates timing uncertainty to transition duration and SNR, and shows
relative uncertainty increasing for shorter exposures. The paper is useful
for method selection; it does not validate our aperture model or demonstrate
that its quoted accuracy is obtained with this hardware.

[Masson et al., Device and algorithms for camera timing evaluation (2014)](https://corp.dxomark.com/wp-content/uploads/2017/11/2014_TimingMeasurements_2014-01-31_wCopyright.pdf)
demonstrate synchronized LED rows that encode time into camera images. This
is an independent timing observable, with explicit operating assumptions.
It does not directly recover shutter exposure from our audio arrays, and
applying image-based methods to an FE2 requires a suitable recording path.

[ESO's detector monitoring plan, section 5.4.18](https://www.eso.org/~qc/docu/detmon/Detector_monitoring_plan_last.pdf)
describes inferring spatial shutter-delay patterns by comparing flat-field
exposures with different opening/closing counts. Its assumptions include
unchanged flux and repeatable per-cycle delay. It is a proposed monitoring
procedure, not a universal correction for fast photographic curtain slits.

The [STScI WFC3 handbook, section 5.6.4](https://hst-docs.stsci.edu/wfc3dhb/chapter-5-wfc3-uvis-sources-of-error/5-6-generic-detector-and-camera-properties)
reports measured shutter shading, timing accuracy and repeatability separately.
That operational example supports distinguishing spatial uniformity from total
exposure accuracy. Its shutter times and mechanism are different from the FE2's;
its numerical corrections must not be reused here.

## 10. What this changes in our interpretation

The following conclusions combine those methods with our own capture audits:

1. **Transient edges is not established as universally most accurate.** It is
   the current default and agrees with the same-batch response experiment at
   1/1000 and 1/2000. That agreement is not independent calibration.
2. **The 1/4000 fitted numbers remain conditional.** Reference changes move
   CH1 substantially and some fits predict nonphysical reconstructed light.
   An authoritative corrected result must reject or resolve those failures.
3. **Movement is only one hypothesis.** The reported resistor adjustments and
   the actual 3 V/load circuit provide another reason shapes may change with
   signal strength. The available recordings do not isolate these mechanisms.
4. **Brightness can cancel in a properly normalized linear area measurement.**
   It cannot be assumed to cancel when the full-light reference comes from a
   different illumination state or the receiver is nonlinear.
5. **C-C arithmetic and exposure calibration are separate.** Correct spacing
   conversion does not remove channel-dependent marker delays or establish
   local aperture velocity. It also does not independently validate a camera
   preset's expected travel time.
6. **The integration route deserves priority over further unconstrained fitting.**
   This is my revised engineering recommendation after the broader review.
   It is supported by the standard's exposure definition and the practical
   remembered-reference implementation, with the circuit qualifications above.

I found no directly verified publication demonstrating accurate exposure for
this specific SFH309/load/audio assembly using only its 2 mm hole and C-C
spacing. This is a limit of the evidence found, not a claim that correction
with those holes is impossible.

## 11. A concrete validation sequence for this rig

This is our proposed experiment, synthesized from the findings above. It is
not a test protocol copied from a standard and has not yet been run.

1. **Document the actual two circuits.** Record collector resistors, series
   resistors, coupling capacitors, supply voltage and audio input settings.
   Check collector DC voltage in dark and full illumination with the camera
   held open. Verify operating headroom and linearity across the light range.
2. **Characterize temporal response independently of the shutter.** Illuminate
   each complete sensor through its installed opening with known-duration
   optical pulses, for example 125, 250, 500 and 1000 us. Independently check
   the optical pulse timing; an electrical LED drive edge is not automatically
   an exact optical edge. Repeat at several brightnesses and compare normalized
   area, delay and rise/fall profiles.
3. **Fix the camera, sensors and light mechanically.** Keep the source stable,
   sufficiently uniform and free of relevant modulation. Collect a full-open
   reference for each channel before and after each fast-speed sequence.
   A source monitor can help detect intensity drift, while rigid positioning
   addresses changes that such a monitor cannot see.
4. **Validate the area path.** With a DC receiver, integrate dark-subtracted
   light and normalize to the corresponding full-open reference. With audio,
   first validate the complete transfer-function inverse and initial-state
   handling against the known pulses. Check that recovered area does not change
   materially with arbitrary integration-window choices.
5. **Characterize spatial response if point timing or waveform fitting is
   required.** A controlled edge or spot scan can measure the assembled
   footprint. A stationary known-duration light pulse characterizes temporal
   electronics but does not alone characterize a moving curtain across 2 mm.
6. **Cross-check shutter results independently.** Compare accepted light-area
   exposure with calibrated edge/spot timing or a suitable separate tester.
   Repeat a controlled sensor flip with the same light and unchanged circuit.
   Do not calibrate the answer by forcing an unverified camera to its label.
7. **Use an uncertainty budget.** Include reference stability, response changes,
   baseline/inversion error, sample timing, noise and repeatability. Sensor
   agreement and a high correlation/SNR are diagnostics, not substitutes.

At 96 kHz, the sample interval is 10.417 us. Nominal 1/1000, 1/2000 and 1/4000
span 96, 48 and 24 samples. The historical 100-sample timebase criterion would
correspond to 400,000 timing ticks/s for 250 us. That clause concerns a
time-interval meter's clock, not necessarily its ADC rate. Our edge indices
currently use the audio sample grid. Interpolation can estimate within that
grid under a model, but does not create new bandwidth or verify accuracy.
Conversely, 24 samples is not a proof that useful integrated exposure is
impossible: that must be determined by calibration and error analysis.

The research itself did not change the live algorithm. A subsequent
implementation adds an optional reference-based light-area mode; it does not
replace the need for the independent experiments above. The [existing aperture
derivation](APERTURE_CORRECTION.md) and local debug reports describe the earlier
mathematical and capture-based experiments.
