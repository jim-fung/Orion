# Backlog Roadmap

## Wave 1 — Integrity and parsing foundation

Prioritize ORION-0031 through ORION-0036, ORION-0085 through ORION-0090, ORION-0039, ORION-0040, and ORION-0110. Outcomes: canonical BLOBFILE semantics, strict parsers/manifests, correct plan binding, checked memory access, and path confinement.

## Wave 2 — Lifecycle and transactional safety

Prioritize ORION-0043 through ORION-0054, ORION-0055/0056, ORION-0061, ORION-0068, and ORION-0078. Outcomes: leases, single-flight compile, state synchronization, durable checkpoints, globally atomic training updates, transactional inference fallback, and deterministic worker cleanup.

## Wave 3 — Independent correctness evidence

Prioritize ORION-0060, ORION-0065, ORION-0067, ORION-0076, ORION-0084, ORION-0099, and ORION-0101/0102. Outcomes: independent CPU/ANE/training/depth oracles and explicit capability coverage.

## Wave 4 — Operational path hardening

Prioritize ORION-0007 through ORION-0011, ORION-0061 through ORION-0066, ORION-0073 through ORION-0078, and ORION-0091 through ORION-0096. Outcomes: strict CLI/protocol behavior, plan-aware serving, fail-closed normalization, verified assets, and current model identities.

## Wave 5 — Benchmarks and observability

Prioritize ORION-0103 through ORION-0108 after correctness gates. Historical performance values are not release gates until reproduced under the provenance schema.

## Wave 6 — Documentation, compatibility, and release

Prioritize ORION-0113 through ORION-0120. Outcomes: normative contracts, honest support boundary, CI, compatibility matrix, and reproducible release artifacts.

## Critical dependency chain

`ORION-0031 → ORION-0032/0033 → ORION-0034/0035/0036 → ORION-0073 → ORION-0074/0076 → ORION-0108 → release evidence`

`ORION-0039/0040 → ORION-0043/0045 → ORION-0051 → ORION-0067 → training qualification`
