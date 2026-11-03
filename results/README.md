# Saved V1/V2 aggregates

These are byte-identical aggregate tables from the original completed runs:

- `v1_summary.csv`: V1 TEST, 240-second enrollment, four fixed starts per duration.
- `v2_static_summary.csv`: V2 static TEST, 30/240-second enrollment and every
  nonoverlapping query interval.
- `v2_stream_summary.csv`: balanced chronological streams under three policies.

Rates are fractions, not percentages. These tables use different grids and cannot
be combined into one accuracy result. No participant-level score rows are tracked.
`artifacts/frozen_runs.json` pins the external files used by the verification tool;
`artifacts/SOURCE_ORIGIN.json` records original hashes for these CSVs.
Hybrid and E/L/F results are explicitly outside this subset.
