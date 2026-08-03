# Plan: create a comprehensive tracked backlog system

I will create a new repository-local `backlog/` directory containing a master index, conventions, and **21 separate facet backlogs**. The backlogs will be seeded from current source at `3e29615`, the verified `reports/CODEBASE_ANALYSIS.md`, current tests, and the historical reports—while clearly separating current findings from historical or external VoXtream obligations.

## 1. Backlog conventions

### IDs

- Use a new global, stable namespace: `ORION-0001`, `ORION-0002`, …
- Do not reuse historical `T001–T163` IDs.
- Do not issue new `ORX-###`, `G008`, `G009`, or `VXANE-G008` IDs; those remain external/historical references.
- IDs remain stable if tasks move, split, are rejected, or become superseded.

### Task fields

Every task will contain:

- ID and outcome-oriented title
- Status (`proposed` initially)
- Priority (`P0`–`P3`)
- Size (`XS`–`XL`)
- Dependencies and related tasks
- External references/gates where applicable
- Source evidence with repository-relative paths
- Problem and user/system consequence
- In-scope and out-of-scope boundaries
- Detailed, independently testable acceptance criteria (`AC-1`, `AC-2`, …)
- Verification plan split into portable, asset-dependent, ANE, numerical, performance, security, or documentation evidence as applicable
- Required capabilities/assets
- Rollback or compatibility notes where relevant

### Priority definitions

- **P0:** silent corruption, memory/lifecycle safety, security boundary, or invalid product claim
- **P1:** required correctness/reliability/evidence for a supported workflow
- **P2:** important maintainability, performance, reproducibility, or usability work
- **P3:** cleanup, optional product work, or deferred research

## 2. Control files

Create:

- `backlog/README.md` — workflow, status transitions, priority/size definitions, completion rules
- `backlog/INDEX.md` — canonical cross-backlog registry and dependency summary
- `backlog/TEMPLATE.md` — reusable extensive task template
- `backlog/SCHEMA.md` — controlled fields, evidence classes, acceptance-criteria rules
- `backlog/LEGACY_IDS.md` — historical `T###`, external `ORX-###`, and G008/G009 mappings
- `backlog/GATES.md` — qualification gates and what evidence is still required
- `backlog/ROADMAP.md` — recommended execution waves and critical dependency chain
- `backlog/OUT_OF_SCOPE.md` — explicit non-goals and research spikes that need decisions first

## 3. Separate facet backlogs

Create `backlog/areas/` with these 21 extensive files:

1. `01-repository-build-toolchain.md`
2. `02-cli-product-operations.md`
3. `03-module-architecture-public-apis.md`
4. `04-compiler-ir-builder-validation.md`
5. `05-compiler-passes-frontends-codegen.md`
6. `06-ane-abi-binding-plans.md`
7. `07-private-ane-runtime-iosurface.md`
8. `08-program-cache-lifecycle-errors.md`
9. `09-checkpoints-delta-patching-lora.md`
10. `10-gpt2-cpu-inference.md`
11. `11-gpt2-ane-inference.md`
12. `12-stories110m-training.md`
13. `13-voxtream-depth-integration.md`
14. `14-tokenizers.md`
15. `15-model-assets-conversion-data.md`
16. `16-testing-correctness-oracles.md`
17. `17-benchmarking-profiling-observability.md`
18. `18-security-reliability-portability.md`
19. `19-documentation-maintainability.md`
20. `20-experiments-archive-retirement.md`
21. `21-macapp-packaging-releases.md`

Each area file will include:

- facet scope and current-state summary
- risks and architectural constraints
- an extensive task table
- full task sections with detailed acceptance criteria
- dependencies on tasks in other area files
- required evidence and hardware/asset profiles
- sequencing notes

## 4. Backlog breadth

I will seed roughly **100–130 actionable tasks**, enough to cover every active and inactive facet without turning each minor observation into noise. The most detailed P0/P1 tasks will include 5–10 acceptance criteria; P2/P3 tasks will generally include 3–6.

The highest-priority clusters will include:

- canonical BLOBFILE payload/hash semantics
- aggregate-first-chunk and overlap correctness
- program/plan binding during eval
- raw depth command migration to plan-aware compile/eval
- host final-RMSNorm fail-closed behavior
- cache lease/lifecycle concurrency
- checked IOSurface API migration
- globally transactional Stories weight generations
- independent forward/backward/Adam oracles
- GPT-2 context/KV bounds and transactional fallback
- strict asset manifests/parsers and converter reproducibility
- strict CLI/depth protocol validation
- capability-aware test profiles and SKIP accounting
- current end-to-end seqpack parity/performance evidence
- private-API compatibility, isolation, and release boundaries

## 5. Cross-file consistency

- Allocate IDs globally and monotonically across all area files.
- Populate `INDEX.md` with every task, area, priority, status, size, dependencies, and external refs.
- Add a dependency-wave roadmap:
  1. integrity and parsing foundation
  2. lifecycle and memory safety
  3. numerical correctness/oracles
  4. operational depth and training hardening
  5. capability-aware tests and current benchmarks
  6. reproducibility, documentation, and release work
- Use repository-relative source references.
- Mark historical benchmark evidence and external VoXtream requirements explicitly rather than presenting them as current source proof.

## 6. Validation after creation

I will run read-only consistency checks to verify:

- every `ORION-####` ID is unique
- IDs in the index match IDs in area files
- all dependencies reference existing tasks
- no historical/external ID is accidentally allocated as a local task
- every task has acceptance criteria and source evidence
- Markdown fences and headings are balanced
- all 21 facet files and control files are linked from `backlog/README.md`
- `git diff --check` passes

No application source code will be changed, and no ANE tests or benchmarks are needed for creating the backlog.