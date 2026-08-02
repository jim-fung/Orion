# ANE ABI and Binding Plans Backlog

## Facet scope

I/O allocation/order, BLOBFILE layout, identity/hash semantics, aggregate chunks, plan/program binding, and plan-aware runtime gates.

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
| ORION-0031 | Define one canonical BLOBFILE payload and hash model | P0 | L | — |
| ORION-0032 | Support aggregate first chunks at offset 64 | P0 | M | ORION-0031 |
| ORION-0033 | Correct aggregate overlap coordinates and overflow checks | P0 | M | ORION-0031 |
| ORION-0034 | Rebind patched plans with per-weight payload hashes | P0 | M | ORION-0031; ORION-0032 |
| ORION-0035 | Bind eval plans to compiled program identity | P0 | L | ORION-0031 |
| ORION-0036 | Unify legacy and aggregate offset validation profiles | P1 | L | ORION-0031; ORION-0032; ORION-0033 |

## Sequencing guidance

Work P0 integrity and safety tasks before enabling broader operational or release claims. P1 evidence and lifecycle work follows; P2/P3 cleanup should not mask unresolved correctness gaps.

---

## ORION-0031 — Define one canonical BLOBFILE payload and hash model

**Status:** proposed
**Priority:** P0 — silent-correctness, security, memory/lifecycle, or claim-integrity blocker
**Size:** L
**Dependencies:** None
**Required capabilities/assets:** Apple Silicon + supported ANE hardware for hardware criteria; portable tests otherwise

### Problem and consequence

Eliminate incompatible whole-file and payload-range digest semantics. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `compiler/binding_plan.c`
- `core/ane_runtime.m`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** One shared helper defines header, payload range, bounds, and digest for legacy and aggregate weights across generation, validation, and patch.
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

## ORION-0032 — Support aggregate first chunks at offset 64

**Status:** proposed
**Priority:** P0 — silent-correctness, security, memory/lifecycle, or claim-integrity blocker
**Size:** M
**Dependencies:** ORION-0031
**Required capabilities/assets:** Apple Silicon + supported ANE hardware for hardware criteria; portable tests otherwise

### Problem and consequence

Make runtime validation match filesystem-side aggregate support. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `compiler/binding_plan.c`
- `core/ane_runtime.m`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** A shared path may contain first/middle/last chunks while true legacy duplicates remain rejected.
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

## ORION-0033 — Correct aggregate overlap coordinates and overflow checks

**Status:** proposed
**Priority:** P0 — silent-correctness, security, memory/lifecycle, or claim-integrity blocker
**Size:** M
**Dependencies:** ORION-0031
**Required capabilities/assets:** Apple Silicon + supported ANE hardware for hardware criteria; portable tests otherwise

### Problem and consequence

Validate the exact payload ranges consumed at runtime. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `compiler/binding_plan.c`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** Adjacent chunks pass; one-byte overlaps, header collisions, wraparound, and out-of-file ranges fail.
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

## ORION-0034 — Rebind patched plans with per-weight payload hashes

**Status:** proposed
**Priority:** P0 — silent-correctness, security, memory/lifecycle, or claim-integrity blocker
**Size:** M
**Dependencies:** ORION-0031, ORION-0032
**Required capabilities/assets:** Apple Silicon + supported ANE hardware for hardware criteria; portable tests otherwise

### Problem and consequence

Stop assigning one physical-file hash to every aggregate weight. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

### Acceptance criteria

- [ ] **AC-1 — Outcome:** Patched aggregate plans carry correct per-chunk hashes and validate through the same path as fresh plans.
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

## ORION-0035 — Bind eval plans to compiled program identity

**Status:** proposed
**Priority:** P0 — silent-correctness, security, memory/lifecycle, or claim-integrity blocker
**Size:** L
**Dependencies:** ORION-0031
**Required capabilities/assets:** Apple Silicon + supported ANE hardware for hardware criteria; portable tests otherwise

### Problem and consequence

Reject valid but foreign same-shaped plans before private evaluation. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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
- `compiler/binding_plan.h`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** Plan-aware eval requires program plan identity and detects MIL, graph provenance, weight, binding, or allocation tampering.
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

## ORION-0036 — Unify legacy and aggregate offset validation profiles

**Status:** proposed
**Priority:** P1 — required correctness, reliability, or evidence for a supported workflow
**Size:** L
**Dependencies:** ORION-0031, ORION-0032, ORION-0033
**Required capabilities/assets:** Apple Silicon + supported ANE hardware for hardware criteria; portable tests otherwise

### Problem and consequence

Remove the offset-64 blanket rule that forces aggregate paths to bypass normal validation. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `compiler/pass_ane_validate.c`
- `apps/cli/commands/depth_serve.m`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** Versioned layout profiles allow valid aggregate offsets and reject malformed layouts before codegen/compile.
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
