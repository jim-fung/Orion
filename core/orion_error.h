#ifndef ORION_ERROR_H
#define ORION_ERROR_H

///
/// Orion native failure taxonomy (ORX-400).
///
/// Every native failure maps to exactly one of these codes. Each code has a
/// stable name and a PRD-mapped recommended action. The recommended action is
/// one of: "continue", "retry", "worker_replace", "mlx_recovery",
/// "request_fail", or "cancel". See orion_error_recommended_action().
///
/// The integer values are a stability contract: VoXtream mirrors them as a
/// Python IntEnum (voxtream.backends.orion.errors.OrionError). Do NOT renumber
/// existing codes; only append new ones.
///
/// Budget notes (ORX-408): ORION_ERR_BUDGET_EXHAUSTED fails closed. The native
/// compile counter (core/ane_runtime.m::g_compile_count) is monotonic per
/// process; the durable view lives in core/orion_compile_ledger.h.
///

#define ORION_COMPILE_THRESHOLD_DEFAULT 90   // PRD safety threshold per process
#define ORION_COMPILE_EMPIRICAL_LIMIT  119   // observed ANE limit per process

typedef enum {
    ORION_OK                   = 0,  // success, no action
    ORION_ERR_COMPILE_FAILED   = 1,  // graph/plan/weight compile error
    ORION_ERR_EVAL_FAILED      = 2,  // ANE eval returned false
    ORION_ERR_BUDGET_EXHAUSTED = 3,  // compile count at threshold (fail-closed)
    ORION_ERR_SURFACE_INVALID  = 4,  // IOSurface validation failed
    ORION_ERR_PLAN_INVALID     = 5,  // binding plan validation failed
    ORION_ERR_TIMEOUT          = 6,  // eval watchdog fired
    ORION_ERR_WORKER_DIED      = 7,  // worker process crashed
    ORION_ERR_CANCELLED        = 8,  // clean discard (request cancelled)
} OrionError;

/// Human-readable name for a code (e.g. "ORION_ERR_COMPILE_FAILED").
/// Returns "ORION_UNKNOWN" for values outside the taxonomy.
const char* orion_error_name(OrionError e);

/// PRD-mapped recommended action for a code. Returns one of:
///   "continue", "retry", "worker_replace", "mlx_recovery",
///   "request_fail", "cancel".
/// Returns "request_fail" for values outside the taxonomy (fail closed).
const char* orion_error_recommended_action(OrionError e);

#endif // ORION_ERROR_H
