# Camera preset database (v12)

`camera_presets.csv` is the data source used by `shutter_tester_v12.py`.
Keep it in the same directory as the Python script.

## Columns

- `name` — display/search name; must be unique.
- `reference_distance_mm` — distance over which the stored service travel time was measured. This is **not necessarily the full 36 mm frame width**.
- `x_sync` — reference flash X-sync speed/text.
- `travel_ms` — expected curtain travel time over `reference_distance_mm`.
- `travel_quality` — `factory`, `family`, `estimate`, or `unknown`.
- `service_travel_tolerance_pct` — symmetric service tolerance in percent, when a manual states it that way.
- `service_travel_min_ms`, `service_travel_max_ms` — explicit service range; preferred over converting a manual range to percent. Either side may be blank for a one-sided limit.
- `service_mismatch_max_ms` — maximum permitted absolute difference between 1st- and 2nd-curtain travel times, only when explicitly documented.
- `note` — provenance/measurement caveats.
- `source_url` — source used for the service value.

## Auto tolerance behavior

The tester uses explicit `service_travel_min_ms` / `service_travel_max_ms` first, then `service_travel_tolerance_pct`. If neither is present it uses the generic ±5% fallback. For curtain mismatch it uses `service_mismatch_max_ms` when available; otherwise Auto uses the generic 5% mismatch limit.

Do not infer a travel-time tolerance from shutter-speed exposure tolerance, X-sync speed, or side-to-side exposure-uniformity limits. If a service manual gives a travel time but the measurement distance is uncertain, leave `travel_ms` blank and document the value in `note` instead.
