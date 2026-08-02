# Testing and Correctness Oracles Backlog

## Facet scope

Capability profiles, PASS/FAIL/SKIP semantics, independent references, plan regressions, sanitizers, fuzzing, and hardware matrix.

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
| ORION-0091 | Report PASS FAIL and SKIP separately | P1 | L | ORION-0003 |
| ORION-0092 | Split tests into explicit capability profiles | P1 | L | ORION-0091; ORION-0097 |
| ORION-0093 | Add portable regressions for binding-plan gaps | P0 | L | ORION-0031; ORION-0032; ORION-0033; ORION-0034; ORION-0035 |
| ORION-0094 | Add sanitizer and fuzzing profiles | P1 | XL | ORION-0004; ORION-0086 |
| ORION-0095 | Implement or retire excluded tests and weak goldens | P2 | M | ORION-0060 |
| ORION-0096 | Establish supported hardware and OS qualification matrix | P1 | L | ORION-0037; ORION-0098 |

## Sequencing guidance

Work P0 integrity and safety tasks before enabling broader operational or release claims. P1 evidence and lifecycle work follows; P2/P3 cleanup should not mask unresolved correctness gaps.

---

## ORION-0091 — Report PASS FAIL and SKIP separately

**Status:** proposed
**Priority:** P1 — required correctness, reliability, or evidence for a supported workflow
**Size:** L
**Dependencies:** ORION-0003
**Required capabilities/assets:** Portable/local unless a criterion explicitly requires assets or ANE hardware

### Problem and consequence

Stop counting skipped hardware/export coverage as passes. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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
- `tests/`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** Harness preserves logs and reports assertion/skip reason/duration; required profiles fail when required coverage skips.
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

## ORION-0092 — Split tests into explicit capability profiles

**Status:** proposed
**Priority:** P1 — required correctness, reliability, or evidence for a supported workflow
**Size:** L
**Dependencies:** ORION-0091, ORION-0097
**Required capabilities/assets:** Portable/local unless a criterion explicitly requires assets or ANE hardware

### Problem and consequence

Separate portable, CPU, conversion, asset, ANE, VoXtream, endurance, and benchmark execution. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- [ ] **AC-1 — Outcome:** Each target declares prerequisites and default test semantics are unambiguous.
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

## ORION-0093 — Add portable regressions for binding-plan gaps

**Status:** proposed
**Priority:** P0 — silent-correctness, security, memory/lifecycle, or claim-integrity blocker
**Size:** L
**Dependencies:** ORION-0031, ORION-0032, ORION-0033, ORION-0034, ORION-0035
**Required capabilities/assets:** Portable/local unless a criterion explicitly requires assets or ANE hardware

### Problem and consequence

Prevent hash/range/first-chunk/foreign-plan regressions. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `tests/test_ane_passes.m`
- `tests/test_plan_weight_dict.m`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** All source-verified integrity discrepancies have direct negative tests, ANE only where unavoidable.
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

## ORION-0094 — Add sanitizer and fuzzing profiles

**Status:** proposed
**Priority:** P1 — required correctness, reliability, or evidence for a supported workflow
**Size:** XL
**Dependencies:** ORION-0004, ORION-0086
**Required capabilities/assets:** Portable/local unless a criterion explicitly requires assets or ANE hardware

### Problem and consequence

Exercise parsers, tensors, checkpoints, tokenizers, data, protocol, and CPU kernels under hostile inputs. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `tests/`
- `Makefile`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** ASan/UBSan jobs and retained fuzz corpora run for documented durations with reproducible failures.
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

## ORION-0095 — Implement or retire excluded tests and weak goldens

**Status:** proposed
**Priority:** P2 — important maintainability, performance, reproducibility, or usability work
**Size:** M
**Dependencies:** ORION-0060
**Required capabilities/assets:** Portable/local unless a criterion explicitly requires assets or ANE hardware

### Problem and consequence

Remove misleading stubs/unwired checks. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `tests/test_program_swap.m`
- `tests/test_infer_golden.json`
- `tests/test_weight_convert.py`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** Each artifact is integrated with assertions or removed with rationale; benchmarks do not count as correctness tests.
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

## ORION-0096 — Establish supported hardware and OS qualification matrix

**Status:** proposed
**Priority:** P1 — required correctness, reliability, or evidence for a supported workflow
**Size:** L
**Dependencies:** ORION-0037, ORION-0098
**Required capabilities/assets:** Portable/local unless a criterion explicitly requires assets or ANE hardware

### Problem and consequence

Tie private-runtime claims to exact SoC/macOS/compiler/assets. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `docs/`
- `tests/`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** Matrix records required suites and OS updates trigger explicit qualification rather than inferred support.
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
