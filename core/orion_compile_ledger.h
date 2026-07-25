#ifndef ORION_COMPILE_LEDGER_H
#define ORION_COMPILE_LEDGER_H

#include <stdio.h>

///
/// Durable compile-budget ledger (ORX-408).
///
/// The native runtime's orion_compile_count() (core/ane_runtime.m) is a
/// process-static monotonic counter: it survives clear_cache divergence and
/// records nothing about planned/cached/failed compiles. This ledger is the
/// structured view the PRD requires: planned, cached, successful, failed, and
/// remaining compiles are visible, and the threshold fails closed.
///
/// Pure C: no ANE calls, no Objective-C, no allocation beyond the struct.
///

/// Per-run compile-budget accounting.
typedef struct {
    int planned;          // declared before a run (budget intent)
    int cached;           // cache hits (no fresh compile consumed)
    int successful;       // fresh compiles that succeeded
    int failed;           // fresh compiles that failed
    int threshold;        // safety threshold (default ORION_COMPILE_THRESHOLD_DEFAULT)
    int empirical_limit;  // observed ANE limit (default ORION_COMPILE_EMPIRICAL_LIMIT)
} OrionCompileLedger;

/// Initialize a ledger. `planned` is the declared run intent; `threshold` is
/// the fail-closed safety threshold (use ORION_COMPILE_THRESHOLD_DEFAULT for
/// the PRD default). `empirical_limit` is pinned to ORION_COMPILE_EMPIRICAL_LIMIT.
OrionCompileLedger orion_ledger_init(int planned, int threshold);

/// Record one compile attempt. A cached hit (cached=true) never advances
/// `successful` or `failed`. A non-cached success advances `successful`; a
/// non-cached failure advances `failed` only (it does not consume a success
/// slot, so it does not advance `successful`).
void orion_ledger_record_compile(OrionCompileLedger* ledger, int success, int cached);

/// Remaining fresh-compile budget: threshold - successful, clamped to >= 0.
int orion_ledger_remaining(const OrionCompileLedger* ledger);

/// Fail-closed test: true once `successful` reaches `threshold`.
int orion_ledger_exhausted(const OrionCompileLedger* ledger);

/// Write a JSON summary to `out` (one object, no trailing newline).
/// Returns the number of characters written, or a negative value on error.
int orion_ledger_to_dict(const OrionCompileLedger* ledger, FILE* out);

#endif // ORION_COMPILE_LEDGER_H
