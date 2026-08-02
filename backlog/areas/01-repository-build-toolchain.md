# Repository, Build, and Toolchain Backlog

## Facet scope

Makefile, root build behavior, dependency tracking, developer setup, generated inventories, and clean-checkout reproducibility.

## Current-state framing

This backlog is seeded from source revision `3e29615` and the verified analysis in `reports/CODEBASE_ANALYSIS.md`. Tasks are proposals, not proof that the cited finding remains unchanged after future edits. Hardware and historical claims require fresh evidence before closure.

## Architectural constraints

- Preserve the ANE I/O, allocation, ordering, and BLOBFILE invariants documented by the current source until a versioned replacement is approved.
- Prefer fail-closed behavior for silent-correctness and integrity risks.
- Keep portable verification separate from ANE-, asset-, and external-export-dependent evidence.
- External `ORX-*`, `G008`, `G009`, and `VXANE-G008` references are gates or provenance, not local task IDs.

## Task summary

| ID | Task | Priority | Size | Blocked by |
|---|---|---|---|---|
| ORION-0001 | Correct and generate build inventories | P2 | S | — |
| ORION-0002 | Add compiler-generated header dependencies | P1 | M | — |
| ORION-0003 | Define capability-aware build profiles | P1 | M | — |
| ORION-0004 | Add strict warning and sanitizer build modes | P1 | M | ORION-0002 |
| ORION-0005 | Create clean-checkout developer bootstrap | P2 | M | ORION-0003 |
| ORION-0006 | Add repository hygiene validation | P2 | S | — |

## Sequencing guidance

Work P0 integrity and safety tasks before enabling broader operational or release claims. P1 evidence and lifecycle work follows; P2/P3 cleanup should not mask unresolved correctness gaps.

---

## ORION-0001 — Correct and generate build inventories

**Status:** proposed
**Priority:** P2 — important maintainability, performance, reproducibility, or usability work
**Size:** S
**Dependencies:** None
**Required capabilities/assets:** Portable/local unless a criterion explicitly requires assets or ANE hardware

### Problem and consequence

Replace stale hand-maintained source and test counts with generated inventory output. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

### Scope

**In scope**
- Implement the behavior described by the outcome below in the cited facet.
- Add positive, negative, boundary, and regression evidence appropriate to the risk.
- Update normative documentation or schemas when the contract changes.

**Out of scope**
- Unrelated model architecture changes or performance tuning.
- Treating historical benchmark prose as proof of current behavior.
- Expanding support to unqualified operating systems, devices, or hostile multi-tenant use.

### Source evidence

- `Makefile`
- `reports/CODEBASE_ANALYSIS.md`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** The Makefile and documentation derive source/test counts from authoritative lists and fail validation when they diverge.
- [ ] **AC-2 — Fail closed:** Invalid, missing, truncated, mismatched, out-of-range, or incompatible inputs relevant to this task are rejected before externally visible state is committed.
- [ ] **AC-3 — Regression coverage:** Existing supported behavior has an automated regression test, and the test fails against a fixture reproducing the original gap.
- [ ] **AC-4 — Diagnostics:** Failures identify the operation and affected object (such as path, tensor, layer, binding, option, or profile) without requiring private-framework log parsing.
- [ ] **AC-5 — Evidence:** Completion records the exact source revision, commands, PASS/FAIL/SKIP counts, required assets, and hardware/OS/compiler identity for non-portable verification.
- [ ] **AC-6 — Compatibility:** Any compatibility, migration, fallback, or rollback behavior is documented; no silent fallback weakens correctness.

### Verification plan

| Layer | Required verification | Completion evidence |
|---|---|---|
| Portable/source | Focused unit or static test plus the relevant portable profile | Command, exit status, assertion count, and log |
| Negative/boundary | Malformed and boundary fixtures for the original failure mode | Named fixtures and expected errors |
| Integration | Exercise the owning command/module through its public contract | End-to-end result and cleanup state |
| Hardware/numerical | When applicable, compile/eval on a qualified Apple Silicon profile and compare with an independent reference | SoC, macOS, compiler profile, assets, tolerances, observed values |
| Documentation | Validate normative docs/schema/help against implementation | Link or generated-doc check |

### Dependencies and rollout

- Blocking dependencies must be complete before this task is marked `ready` unless the task is explicitly split.
- Preserve a known-good path until replacement evidence passes.
- Close only after all mandatory acceptance criteria are checked and residual limitations are listed.

---

## ORION-0002 — Add compiler-generated header dependencies

**Status:** proposed
**Priority:** P1 — required correctness, reliability, or evidence for a supported workflow
**Size:** M
**Dependencies:** None
**Required capabilities/assets:** Portable/local unless a criterion explicitly requires assets or ANE hardware

### Problem and consequence

Prevent stale object reuse after header or model-config changes. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

### Scope

**In scope**
- Implement the behavior described by the outcome below in the cited facet.
- Add positive, negative, boundary, and regression evidence appropriate to the risk.
- Update normative documentation or schemas when the contract changes.

**Out of scope**
- Unrelated model architecture changes or performance tuning.
- Treating historical benchmark prose as proof of current behavior.
- Expanding support to unqualified operating systems, devices, or hostile multi-tenant use.

### Source evidence

- `Makefile`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** C and Objective-C rules emit and include dependency files; a header-only change rebuilds every affected object.
- [ ] **AC-2 — Fail closed:** Invalid, missing, truncated, mismatched, out-of-range, or incompatible inputs relevant to this task are rejected before externally visible state is committed.
- [ ] **AC-3 — Regression coverage:** Existing supported behavior has an automated regression test, and the test fails against a fixture reproducing the original gap.
- [ ] **AC-4 — Diagnostics:** Failures identify the operation and affected object (such as path, tensor, layer, binding, option, or profile) without requiring private-framework log parsing.
- [ ] **AC-5 — Evidence:** Completion records the exact source revision, commands, PASS/FAIL/SKIP counts, required assets, and hardware/OS/compiler identity for non-portable verification.
- [ ] **AC-6 — Compatibility:** Any compatibility, migration, fallback, or rollback behavior is documented; no silent fallback weakens correctness.

### Verification plan

| Layer | Required verification | Completion evidence |
|---|---|---|
| Portable/source | Focused unit or static test plus the relevant portable profile | Command, exit status, assertion count, and log |
| Negative/boundary | Malformed and boundary fixtures for the original failure mode | Named fixtures and expected errors |
| Integration | Exercise the owning command/module through its public contract | End-to-end result and cleanup state |
| Hardware/numerical | When applicable, compile/eval on a qualified Apple Silicon profile and compare with an independent reference | SoC, macOS, compiler profile, assets, tolerances, observed values |
| Documentation | Validate normative docs/schema/help against implementation | Link or generated-doc check |

### Dependencies and rollout

- Blocking dependencies must be complete before this task is marked `ready` unless the task is explicitly split.
- Preserve a known-good path until replacement evidence passes.
- Close only after all mandatory acceptance criteria are checked and residual limitations are listed.

---

## ORION-0003 — Define capability-aware build profiles

**Status:** proposed
**Priority:** P1 — required correctness, reliability, or evidence for a supported workflow
**Size:** M
**Dependencies:** None
**Required capabilities/assets:** Portable/local unless a criterion explicitly requires assets or ANE hardware

### Problem and consequence

Separate portable, Apple-Silicon, ANE, asset, and VoXtream build/test requirements. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

### Scope

**In scope**
- Implement the behavior described by the outcome below in the cited facet.
- Add positive, negative, boundary, and regression evidence appropriate to the risk.
- Update normative documentation or schemas when the contract changes.

**Out of scope**
- Unrelated model architecture changes or performance tuning.
- Treating historical benchmark prose as proof of current behavior.
- Expanding support to unqualified operating systems, devices, or hostile multi-tenant use.

### Source evidence

- `Makefile`
- `AGENTS.md`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** Named profiles document prerequisites and never report missing required capabilities as successful coverage.
- [ ] **AC-2 — Fail closed:** Invalid, missing, truncated, mismatched, out-of-range, or incompatible inputs relevant to this task are rejected before externally visible state is committed.
- [ ] **AC-3 — Regression coverage:** Existing supported behavior has an automated regression test, and the test fails against a fixture reproducing the original gap.
- [ ] **AC-4 — Diagnostics:** Failures identify the operation and affected object (such as path, tensor, layer, binding, option, or profile) without requiring private-framework log parsing.
- [ ] **AC-5 — Evidence:** Completion records the exact source revision, commands, PASS/FAIL/SKIP counts, required assets, and hardware/OS/compiler identity for non-portable verification.
- [ ] **AC-6 — Compatibility:** Any compatibility, migration, fallback, or rollback behavior is documented; no silent fallback weakens correctness.

### Verification plan

| Layer | Required verification | Completion evidence |
|---|---|---|
| Portable/source | Focused unit or static test plus the relevant portable profile | Command, exit status, assertion count, and log |
| Negative/boundary | Malformed and boundary fixtures for the original failure mode | Named fixtures and expected errors |
| Integration | Exercise the owning command/module through its public contract | End-to-end result and cleanup state |
| Hardware/numerical | When applicable, compile/eval on a qualified Apple Silicon profile and compare with an independent reference | SoC, macOS, compiler profile, assets, tolerances, observed values |
| Documentation | Validate normative docs/schema/help against implementation | Link or generated-doc check |

### Dependencies and rollout

- Blocking dependencies must be complete before this task is marked `ready` unless the task is explicitly split.
- Preserve a known-good path until replacement evidence passes.
- Close only after all mandatory acceptance criteria are checked and residual limitations are listed.

---

## ORION-0004 — Add strict warning and sanitizer build modes

**Status:** proposed
**Priority:** P1 — required correctness, reliability, or evidence for a supported workflow
**Size:** M
**Dependencies:** ORION-0002
**Required capabilities/assets:** Portable/local unless a criterion explicitly requires assets or ANE hardware

### Problem and consequence

Make warnings and CPU-detectable undefined behavior actionable before hardware execution. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

### Scope

**In scope**
- Implement the behavior described by the outcome below in the cited facet.
- Add positive, negative, boundary, and regression evidence appropriate to the risk.
- Update normative documentation or schemas when the contract changes.

**Out of scope**
- Unrelated model architecture changes or performance tuning.
- Treating historical benchmark prose as proof of current behavior.
- Expanding support to unqualified operating systems, devices, or hostile multi-tenant use.

### Source evidence

- `Makefile`
- `compiler/`
- `core/`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** Strict and sanitizer targets build supported portable code with zero unexpected warnings and execute the declared portable suites.
- [ ] **AC-2 — Fail closed:** Invalid, missing, truncated, mismatched, out-of-range, or incompatible inputs relevant to this task are rejected before externally visible state is committed.
- [ ] **AC-3 — Regression coverage:** Existing supported behavior has an automated regression test, and the test fails against a fixture reproducing the original gap.
- [ ] **AC-4 — Diagnostics:** Failures identify the operation and affected object (such as path, tensor, layer, binding, option, or profile) without requiring private-framework log parsing.
- [ ] **AC-5 — Evidence:** Completion records the exact source revision, commands, PASS/FAIL/SKIP counts, required assets, and hardware/OS/compiler identity for non-portable verification.
- [ ] **AC-6 — Compatibility:** Any compatibility, migration, fallback, or rollback behavior is documented; no silent fallback weakens correctness.

### Verification plan

| Layer | Required verification | Completion evidence |
|---|---|---|
| Portable/source | Focused unit or static test plus the relevant portable profile | Command, exit status, assertion count, and log |
| Negative/boundary | Malformed and boundary fixtures for the original failure mode | Named fixtures and expected errors |
| Integration | Exercise the owning command/module through its public contract | End-to-end result and cleanup state |
| Hardware/numerical | When applicable, compile/eval on a qualified Apple Silicon profile and compare with an independent reference | SoC, macOS, compiler profile, assets, tolerances, observed values |
| Documentation | Validate normative docs/schema/help against implementation | Link or generated-doc check |

### Dependencies and rollout

- Blocking dependencies must be complete before this task is marked `ready` unless the task is explicitly split.
- Preserve a known-good path until replacement evidence passes.
- Close only after all mandatory acceptance criteria are checked and residual limitations are listed.

---

## ORION-0005 — Create clean-checkout developer bootstrap

**Status:** proposed
**Priority:** P2 — important maintainability, performance, reproducibility, or usability work
**Size:** M
**Dependencies:** ORION-0003
**Required capabilities/assets:** Portable/local unless a criterion explicitly requires assets or ANE hardware

### Problem and consequence

Make setup deterministic without assuming ignored local assets. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

### Scope

**In scope**
- Implement the behavior described by the outcome below in the cited facet.
- Add positive, negative, boundary, and regression evidence appropriate to the risk.
- Update normative documentation or schemas when the contract changes.

**Out of scope**
- Unrelated model architecture changes or performance tuning.
- Treating historical benchmark prose as proof of current behavior.
- Expanding support to unqualified operating systems, devices, or hostile multi-tenant use.

### Source evidence

- `README.md`
- `requirements.txt`
- `scripts/`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** A documented bootstrap verifies toolchain and optional assets, performs no implicit network fetch, and reports enabled profiles.
- [ ] **AC-2 — Fail closed:** Invalid, missing, truncated, mismatched, out-of-range, or incompatible inputs relevant to this task are rejected before externally visible state is committed.
- [ ] **AC-3 — Regression coverage:** Existing supported behavior has an automated regression test, and the test fails against a fixture reproducing the original gap.
- [ ] **AC-4 — Diagnostics:** Failures identify the operation and affected object (such as path, tensor, layer, binding, option, or profile) without requiring private-framework log parsing.
- [ ] **AC-5 — Evidence:** Completion records the exact source revision, commands, PASS/FAIL/SKIP counts, required assets, and hardware/OS/compiler identity for non-portable verification.
- [ ] **AC-6 — Compatibility:** Any compatibility, migration, fallback, or rollback behavior is documented; no silent fallback weakens correctness.

### Verification plan

| Layer | Required verification | Completion evidence |
|---|---|---|
| Portable/source | Focused unit or static test plus the relevant portable profile | Command, exit status, assertion count, and log |
| Negative/boundary | Malformed and boundary fixtures for the original failure mode | Named fixtures and expected errors |
| Integration | Exercise the owning command/module through its public contract | End-to-end result and cleanup state |
| Hardware/numerical | When applicable, compile/eval on a qualified Apple Silicon profile and compare with an independent reference | SoC, macOS, compiler profile, assets, tolerances, observed values |
| Documentation | Validate normative docs/schema/help against implementation | Link or generated-doc check |

### Dependencies and rollout

- Blocking dependencies must be complete before this task is marked `ready` unless the task is explicitly split.
- Preserve a known-good path until replacement evidence passes.
- Close only after all mandatory acceptance criteria are checked and residual limitations are listed.

---

## ORION-0006 — Add repository hygiene validation

**Status:** proposed
**Priority:** P2 — important maintainability, performance, reproducibility, or usability work
**Size:** S
**Dependencies:** None
**Required capabilities/assets:** Portable/local unless a criterion explicitly requires assets or ANE hardware

### Problem and consequence

Keep generated binaries, blobs, datasets, checkpoints, and local tooling out of tracked releases. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

### Scope

**In scope**
- Implement the behavior described by the outcome below in the cited facet.
- Add positive, negative, boundary, and regression evidence appropriate to the risk.
- Update normative documentation or schemas when the contract changes.

**Out of scope**
- Unrelated model architecture changes or performance tuning.
- Treating historical benchmark prose as proof of current behavior.
- Expanding support to unqualified operating systems, devices, or hostile multi-tenant use.

### Source evidence

- `.gitignore`
- `Makefile`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** An automated check detects forbidden tracked artifacts and distinguishes tracked fixtures from generated evidence.
- [ ] **AC-2 — Fail closed:** Invalid, missing, truncated, mismatched, out-of-range, or incompatible inputs relevant to this task are rejected before externally visible state is committed.
- [ ] **AC-3 — Regression coverage:** Existing supported behavior has an automated regression test, and the test fails against a fixture reproducing the original gap.
- [ ] **AC-4 — Diagnostics:** Failures identify the operation and affected object (such as path, tensor, layer, binding, option, or profile) without requiring private-framework log parsing.
- [ ] **AC-5 — Evidence:** Completion records the exact source revision, commands, PASS/FAIL/SKIP counts, required assets, and hardware/OS/compiler identity for non-portable verification.
- [ ] **AC-6 — Compatibility:** Any compatibility, migration, fallback, or rollback behavior is documented; no silent fallback weakens correctness.

### Verification plan

| Layer | Required verification | Completion evidence |
|---|---|---|
| Portable/source | Focused unit or static test plus the relevant portable profile | Command, exit status, assertion count, and log |
| Negative/boundary | Malformed and boundary fixtures for the original failure mode | Named fixtures and expected errors |
| Integration | Exercise the owning command/module through its public contract | End-to-end result and cleanup state |
| Hardware/numerical | When applicable, compile/eval on a qualified Apple Silicon profile and compare with an independent reference | SoC, macOS, compiler profile, assets, tolerances, observed values |
| Documentation | Validate normative docs/schema/help against implementation | Link or generated-doc check |

### Dependencies and rollout

- Blocking dependencies must be complete before this task is marked `ready` unless the task is explicitly split.
- Preserve a known-good path until replacement evidence passes.
- Close only after all mandatory acceptance criteria are checked and residual limitations are listed.
