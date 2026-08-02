# Private ANE Runtime and IOSurface Backlog

## Facet scope

Framework compatibility, compile/load/eval/unload, surface layouts, checked copying, overflow safety, and timeout/isolation boundaries.

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
| ORION-0037 | Centralize and verify private API compatibility | P1 | L | ORION-0027 |
| ORION-0038 | Migrate operational paths to checked tensor APIs | P0 | XL | ORION-0035 |
| ORION-0039 | Harden legacy IOSurface helpers pending removal | P0 | L | — |
| ORION-0040 | Centralize overflow-checked tensor allocation arithmetic | P1 | M | ORION-0039 |
| ORION-0041 | Add structured private-runtime error context | P1 | M | ORION-0037 |
| ORION-0042 | Prototype timeout and subprocess isolation policy | P2 | L | ORION-0041 |

## Sequencing guidance

Work P0 integrity and safety tasks before enabling broader operational or release claims. P1 evidence and lifecycle work follows; P2/P3 cleanup should not mask unresolved correctness gaps.

---

## ORION-0037 — Centralize and verify private API compatibility

**Status:** proposed
**Priority:** P1 — required correctness, reliability, or evidence for a supported workflow
**Size:** L
**Dependencies:** ORION-0027
**Required capabilities/assets:** Apple Silicon + supported ANE hardware for hardware criteria; portable tests otherwise

### Problem and consequence

Resolve classes/selectors and profile assumptions in one compatibility table. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `core/ane_runtime.m`
- `docs/ane_api_reference.md`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** Preflight reports missing features with OS/SoC/profile identity before model compilation.
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

## ORION-0038 — Migrate operational paths to checked tensor APIs

**Status:** proposed
**Priority:** P0 — silent-correctness, security, memory/lifecycle, or claim-integrity blocker
**Size:** XL
**Dependencies:** ORION-0035
**Required capabilities/assets:** Apple Silicon + supported ANE hardware for hardware criteria; portable tests otherwise

### Problem and consequence

Eliminate unchecked byte copies and repeated shape arithmetic in supported flows. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `core/iosurface_tensor.m`
- `kernels/`
- `apps/cli/commands/`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** GPT-2, Stories, and depth operations derive capacities from bindings and reject mismatched surfaces before access.
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

## ORION-0039 — Harden legacy IOSurface helpers pending removal

**Status:** proposed
**Priority:** P0 — silent-correctness, security, memory/lifecycle, or claim-integrity blocker
**Size:** L
**Dependencies:** None
**Required capabilities/assets:** Apple Silicon + supported ANE hardware for hardware criteria; portable tests otherwise

### Problem and consequence

Prevent out-of-bounds, overflow, lock failure, and overlapping-copy errors. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `core/iosurface_tensor.m`
- `tests/test_iosurface_tensor_checked.m`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** All helpers return status, check allocation/base/lock/ranges, use safe overlap semantics, and pass sanitizer boundary tests.
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

## ORION-0040 — Centralize overflow-checked tensor allocation arithmetic

**Status:** proposed
**Priority:** P1 — required correctness, reliability, or evidence for a supported workflow
**Size:** M
**Dependencies:** ORION-0039
**Required capabilities/assets:** Apple Silicon + supported ANE hardware for hardware criteria; portable tests otherwise

### Problem and consequence

Use shared checked multiplication/addition for all tensor and workspace sizes. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `core/iosurface_tensor.m`
- `kernels/inference/kv_cache.m`
- `kernels/training/`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** Boundary, negative-to-unsigned, and memory-pressure tests fail without corrupting state.
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

## ORION-0041 — Add structured private-runtime error context

**Status:** proposed
**Priority:** P1 — required correctness, reliability, or evidence for a supported workflow
**Size:** M
**Dependencies:** ORION-0037
**Required capabilities/assets:** Apple Silicon + supported ANE hardware for hardware criteria; portable tests otherwise

### Problem and consequence

Preserve operation, selector, program tag, profile, and underlying error information. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `core/ane_runtime.m`
- `core/orion_error.h`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** Compile/load/eval/unload failures return stable codes plus actionable context without log parsing.
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

## ORION-0042 — Prototype timeout and subprocess isolation policy

**Status:** proposed
**Priority:** P2 — important maintainability, performance, reproducibility, or usability work
**Size:** L
**Dependencies:** ORION-0041
**Required capabilities/assets:** Apple Silicon + supported ANE hardware for hardware criteria; portable tests otherwise

### Problem and consequence

Define recovery from wedged or crashing private calls. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `core/ane_runtime.m`
- `apps/cli/commands/depth_serve.m`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** A documented spike measures worker startup/transfer cost and demonstrates parent-side timeout, cancellation, and replacement.
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
