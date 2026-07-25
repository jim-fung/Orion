// core/orion_error.c — Orion native failure taxonomy (ORX-400).
//
// Pure C: no ANE calls, no Objective-C. Links only Foundation when used from a
// compiler-only test. See core/orion_error.h for the value-stability contract.

#include "core/orion_error.h"

const char* orion_error_name(OrionError e) {
    switch (e) {
        case ORION_OK:                   return "ORION_OK";
        case ORION_ERR_COMPILE_FAILED:   return "ORION_ERR_COMPILE_FAILED";
        case ORION_ERR_EVAL_FAILED:      return "ORION_ERR_EVAL_FAILED";
        case ORION_ERR_BUDGET_EXHAUSTED: return "ORION_ERR_BUDGET_EXHAUSTED";
        case ORION_ERR_SURFACE_INVALID:  return "ORION_ERR_SURFACE_INVALID";
        case ORION_ERR_PLAN_INVALID:     return "ORION_ERR_PLAN_INVALID";
        case ORION_ERR_TIMEOUT:          return "ORION_ERR_TIMEOUT";
        case ORION_ERR_WORKER_DIED:      return "ORION_ERR_WORKER_DIED";
        case ORION_ERR_CANCELLED:        return "ORION_ERR_CANCELLED";
        default:                         return "ORION_UNKNOWN";
    }
}

const char* orion_error_recommended_action(OrionError e) {
    // PRD ORX-400 mapping. The full action space also includes "mlx_recovery",
    // which the host orchestrator applies when worker rotation is exhausted
    // (e.g. after repeated BUDGET_EXHAUSTED / WORKER_DIED on a host whose ANE
    // can no longer serve). No single native error recommends mlx_recovery as
    // its primary action; it is an orchestration-level escalation. Kept in the
    // Python OrionErrorAction enum so the two sides agree on the action space.
    switch (e) {
        case ORION_OK:                   return "continue";
        case ORION_ERR_COMPILE_FAILED:   return "retry";
        case ORION_ERR_EVAL_FAILED:      return "retry";
        case ORION_ERR_BUDGET_EXHAUSTED: return "worker_replace";
        case ORION_ERR_SURFACE_INVALID:  return "retry";
        case ORION_ERR_PLAN_INVALID:     return "request_fail";
        case ORION_ERR_TIMEOUT:          return "worker_replace";
        case ORION_ERR_WORKER_DIED:      return "worker_replace";
        case ORION_ERR_CANCELLED:        return "cancel";
        default:                         return "request_fail";  // fail closed
    }
}
