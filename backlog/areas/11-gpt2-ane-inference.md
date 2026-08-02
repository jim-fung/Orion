# GPT-2 ANE Inference Backlog

## Facet scope

ANE prefill/decode correctness, program population, plan-aware execution, fallback atomicity, buckets, and backend evidence.

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
| ORION-0061 | Make ANE decode fallback transactional | P0 | L | ORION-0056; ORION-0045 |
| ORION-0062 | Move GPT-2 programs to program-bound plan eval | P1 | L | ORION-0035; ORION-0038 |
| ORION-0063 | Use content-derived model and compiler cache identity | P1 | M | ORION-0048 |
| ORION-0064 | Plan compile budgets for buckets and decode programs | P1 | M | ORION-0047; ORION-0063 |
| ORION-0065 | Expand ANE parity across buckets and sustained decode | P1 | L | ORION-0060; ORION-0062 |
| ORION-0066 | Report accurate hybrid backend responsibilities | P2 | S | ORION-0009 |

## Sequencing guidance

Work P0 integrity and safety tasks before enabling broader operational or release claims. P1 evidence and lifecycle work follows; P2/P3 cleanup should not mask unresolved correctness gaps.

---

## ORION-0061 — Make ANE decode fallback transactional

**Status:** proposed
**Priority:** P0 — silent-correctness, security, memory/lifecycle, or claim-integrity blocker
**Size:** L
**Dependencies:** ORION-0056, ORION-0045
**Required capabilities/assets:** Apple Silicon + supported ANE hardware for hardware criteria; portable tests otherwise

### Problem and consequence

Prevent partial ANE KV writes before CPU fallback. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `kernels/inference/decode_ane.m`
- `apps/cli/commands/infer.m`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** Forced failure at every layer yields the same cache/logits as pure CPU and advances length once.
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

## ORION-0062 — Move GPT-2 programs to program-bound plan eval

**Status:** proposed
**Priority:** P1 — required correctness, reliability, or evidence for a supported workflow
**Size:** L
**Dependencies:** ORION-0035, ORION-0038
**Required capabilities/assets:** Apple Silicon + supported ANE hardware for hardware criteria; portable tests otherwise

### Problem and consequence

Ensure every surface and plan belongs to the compiled program. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `kernels/inference/prefill_ane.m`
- `kernels/inference/decode_ane.m`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** Foreign/tampered plans and wrong allocations fail before ANE execution.
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

## ORION-0063 — Use content-derived model and compiler cache identity

**Status:** proposed
**Priority:** P1 — required correctness, reliability, or evidence for a supported workflow
**Size:** M
**Dependencies:** ORION-0048
**Required capabilities/assets:** Apple Silicon + supported ANE hardware for hardware criteria; portable tests otherwise

### Problem and consequence

Stop using display IDs such as base for mutable assets. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `kernels/inference/`
- `core/kernel.m`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** Verified manifest and compiler-profile changes force distinct cache entries.
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

## ORION-0064 — Plan compile budgets for buckets and decode programs

**Status:** proposed
**Priority:** P1 — required correctness, reliability, or evidence for a supported workflow
**Size:** M
**Dependencies:** ORION-0047, ORION-0063
**Required capabilities/assets:** Apple Silicon + supported ANE hardware for hardware criteria; portable tests otherwise

### Problem and consequence

Avoid exceeding configured process budgets during warming. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `kernels/inference/prefill_ane.m`
- `apps/cli/commands/infer.m`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** Runtime reserves projected compiles, reports remaining budget, and refuses unsafe warming before compile.
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

## ORION-0065 — Expand ANE parity across buckets and sustained decode

**Status:** proposed
**Priority:** P1 — required correctness, reliability, or evidence for a supported workflow
**Size:** L
**Dependencies:** ORION-0060, ORION-0062
**Required capabilities/assets:** Apple Silicon + supported ANE hardware for hardware criteria; portable tests otherwise

### Problem and consequence

Cover all supported shapes and long cache evolution. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `tests/test_ane_prefill.m`
- `tests/test_decode_ane_step.m`
- `tests/test_infer_golden_ane.m`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** Pinned references define per-tensor/logit/KV tolerances and exact greedy-token expectations.
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

## ORION-0066 — Report accurate hybrid backend responsibilities

**Status:** proposed
**Priority:** P2 — important maintainability, performance, reproducibility, or usability work
**Size:** S
**Dependencies:** ORION-0009
**Required capabilities/assets:** Apple Silicon + supported ANE hardware for hardware criteria; portable tests otherwise

### Problem and consequence

Remove Full ANE terminology where attention/logits/sampling remain CPU-side. Without a bounded task and explicit evidence, this facet can drift or produce results that are difficult to trust.

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

- `apps/cli/commands/infer.m`
- `README.md`

### Acceptance criteria

- [ ] **AC-1 — Outcome:** CLI/docs enumerate actual CPU/ANE stages and fallback history.
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
