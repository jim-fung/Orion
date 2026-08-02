# Backlog Schema and Evidence Rules

## Identifiers

Local IDs use `ORION-####`, globally monotonic and never reused. Historical `T###` and external `ORX-*`/`G*` identifiers are references only.

## Controlled fields

- Status: `proposed`, `triaged`, `ready`, `in_progress`, `blocked`, `in_review`, `done`, `deferred`, `rejected`, `superseded`.
- Priority: `P0`, `P1`, `P2`, `P3`.
- Size: `XS`, `S`, `M`, `L`, `XL`. XL tasks should normally split before becoming ready.
- Evidence class: `source_verified`, `verified_locally`, `test_assertion`, `historical`, `external_unverified`.

## Acceptance-criteria rules

Each AC must be independently decidable. Numerical ACs state tensor/reference/dtype/norm/tolerance. Performance ACs state workload, host, warmup, sample count, statistic, direction, and threshold. Reliability ACs name the injected failure and required resource state. Documentation ACs name the normative file or generated check.

Avoid “works correctly,” “add tests,” or “all gates pass” without naming the exact behavior and evidence scope.

## Evidence record

Record revision, dirty state, date, command, exit status, PASS/FAIL/SKIP, host/SoC, macOS, compiler profile, assets/manifests/hashes, tolerances, observed values, raw artifact location, and known limitations. Large logs/blobs/checkpoints remain outside this folder; commit compact manifests or reproduction commands.

## Dependency semantics

`Blocked by` means acceptance cannot complete first. Related work is informational and should not be represented as a blocker. Hardware/assets are capabilities, not fake task dependencies.
