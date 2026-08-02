# Orion Engineering Backlog

This directory is the tracked, repository-local backlog for Orion. It covers every active and inactive facet of the codebase at source revision `3e29615`.

## Start here

- [`INDEX.md`](INDEX.md) — canonical task registry
- [`ROADMAP.md`](ROADMAP.md) — dependency waves and recommended execution order
- [`SCHEMA.md`](SCHEMA.md) — controlled fields and evidence rules
- [`TEMPLATE.md`](TEMPLATE.md) — template for future tasks
- [`GATES.md`](GATES.md) — qualification and external gates
- [`LEGACY_IDS.md`](LEGACY_IDS.md) — historical/external identifier mapping
- [`OUT_OF_SCOPE.md`](OUT_OF_SCOPE.md) — non-goals and required research spikes
- [`validate_backlog.py`](validate_backlog.py) — portable backlog consistency validator

## Facet backlogs

1. [`01-repository-build-toolchain.md`](areas/01-repository-build-toolchain.md) — Repository, Build, and Toolchain
2. [`02-cli-product-operations.md`](areas/02-cli-product-operations.md) — CLI Product and Operations
3. [`03-module-architecture-public-apis.md`](areas/03-module-architecture-public-apis.md) — Module Architecture and Public APIs
4. [`04-compiler-ir-builder-validation.md`](areas/04-compiler-ir-builder-validation.md) — Compiler IR, Builder, and Validation
5. [`05-compiler-passes-frontends-codegen.md`](areas/05-compiler-passes-frontends-codegen.md) — Compiler Passes, Frontends, and MIL Codegen
6. [`06-ane-abi-binding-plans.md`](areas/06-ane-abi-binding-plans.md) — ANE ABI and Binding Plans
7. [`07-private-ane-runtime-iosurface.md`](areas/07-private-ane-runtime-iosurface.md) — Private ANE Runtime and IOSurface
8. [`08-program-cache-lifecycle-errors.md`](areas/08-program-cache-lifecycle-errors.md) — Program Cache, Lifecycle, and Errors
9. [`09-checkpoints-delta-patching-lora.md`](areas/09-checkpoints-delta-patching-lora.md) — Checkpoints, Delta Patching, and LoRA
10. [`10-gpt2-cpu-inference.md`](areas/10-gpt2-cpu-inference.md) — GPT-2 CPU Inference
11. [`11-gpt2-ane-inference.md`](areas/11-gpt2-ane-inference.md) — GPT-2 ANE Inference
12. [`12-stories110m-training.md`](areas/12-stories110m-training.md) — Stories110M Training
13. [`13-voxtream-depth-integration.md`](areas/13-voxtream-depth-integration.md) — VoXtream Depth Integration
14. [`14-tokenizers.md`](areas/14-tokenizers.md) — Tokenizers
15. [`15-model-assets-conversion-data.md`](areas/15-model-assets-conversion-data.md) — Model Assets, Conversion, and Data
16. [`16-testing-correctness-oracles.md`](areas/16-testing-correctness-oracles.md) — Testing and Correctness Oracles
17. [`17-benchmarking-profiling-observability.md`](areas/17-benchmarking-profiling-observability.md) — Benchmarking, Profiling, and Observability
18. [`18-security-reliability-portability.md`](areas/18-security-reliability-portability.md) — Security, Reliability, and Portability
19. [`19-documentation-maintainability.md`](areas/19-documentation-maintainability.md) — Documentation and Maintainability
20. [`20-experiments-archive-retirement.md`](areas/20-experiments-archive-retirement.md) — Experiments and Archive Retirement
21. [`21-macapp-packaging-releases.md`](areas/21-macapp-packaging-releases.md) — macOS App, Packaging, and Releases

## Workflow

1. New tasks receive the next unused global `ORION-####` ID. IDs are never reused.
2. `proposed` tasks are triaged into `ready`, `blocked`, `deferred`, or `rejected`.
3. A task is `ready` only when scope, dependencies, acceptance criteria, capabilities, and assets are explicit.
4. Only one owner should mark a task `in_progress`; implementation work should cite the task ID.
5. `done` requires every mandatory acceptance criterion and recorded evidence.
6. Split, superseded, or rejected IDs remain searchable and are never deleted.
7. Validate the registry before submitting backlog changes:
   ```sh
   python3 backlog/validate_backlog.py
   ```
   The validator uses only the Python standard library and exits nonzero for invalid IDs or dependencies, index/area record mismatches, `ready`/`done` tasks with unfinished blockers, or incomplete `done` checklists/evidence.

## Priority

- **P0:** silent corruption, memory/lifecycle safety, security boundary, or invalid product claim
- **P1:** required correctness, reliability, or evidence for a supported workflow
- **P2:** important maintainability, performance, reproducibility, or usability
- **P3:** cleanup, optional product work, or deferred research

## Status

`proposed`, `triaged`, `ready`, `in_progress`, `blocked`, `in_review`, `done`, `deferred`, `rejected`, `superseded`.

## Definition of done

- All ACs are checked.
- Required portable and capability-dependent evidence is recorded.
- PASS/FAIL/SKIP are reported separately.
- Documentation/contracts are updated.
- No fallback silently weakens correctness.
- Remaining limitations and follow-ups are explicit.
