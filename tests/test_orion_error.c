// tests/test_orion_error.c — ORX-400 taxonomy tests (compiler-only, zero ANE).
//
// Asserts every OrionError value has a stable name and a PRD-mapped action, and
// that the action mapping matches the documented recovery policy. Pure C; links
// only Foundation when built (unused — matches the compiler-only test pattern).

#include <stdio.h>
#include <string.h>

#include "core/orion_error.h"

static int g_passed = 0, g_failed = 0;

#define TEST(name) do { \
    printf("  %-50s ", #name); \
    if (test_##name()) { printf("PASS\n"); g_passed++; } \
    else { printf("FAIL\n"); g_failed++; } \
} while(0)

// Every defined code must have a non-empty name that is not the sentinel.
static int test_every_code_has_a_name(void) {
    OrionError codes[] = {
        ORION_OK, ORION_ERR_COMPILE_FAILED, ORION_ERR_EVAL_FAILED,
        ORION_ERR_BUDGET_EXHAUSTED, ORION_ERR_SURFACE_INVALID,
        ORION_ERR_PLAN_INVALID, ORION_ERR_TIMEOUT,
        ORION_ERR_WORKER_DIED, ORION_ERR_CANCELLED,
    };
    for (size_t i = 0; i < sizeof(codes) / sizeof(codes[0]); i++) {
        const char* n = orion_error_name(codes[i]);
        if (!n || n[0] == '\0' || strcmp(n, "ORION_UNKNOWN") == 0) return 0;
    }
    return 1;
}

// Every defined code must have a non-empty recommended action.
static int test_every_code_has_an_action(void) {
    OrionError codes[] = {
        ORION_OK, ORION_ERR_COMPILE_FAILED, ORION_ERR_EVAL_FAILED,
        ORION_ERR_BUDGET_EXHAUSTED, ORION_ERR_SURFACE_INVALID,
        ORION_ERR_PLAN_INVALID, ORION_ERR_TIMEOUT,
        ORION_ERR_WORKER_DIED, ORION_ERR_CANCELLED,
    };
    for (size_t i = 0; i < sizeof(codes) / sizeof(codes[0]); i++) {
        const char* a = orion_error_recommended_action(codes[i]);
        if (!a || a[0] == '\0') return 0;
    }
    return 1;
}

// Unknown codes fail closed: name is the sentinel, action is request_fail.
static int test_unknown_codes_fail_closed(void) {
    if (strcmp(orion_error_name((OrionError)999), "ORION_UNKNOWN") != 0) return 0;
    if (strcmp(orion_error_recommended_action((OrionError)999), "request_fail") != 0) return 0;
    return 1;
}

// The documented PRD recovery policy per code.
static int test_action_mapping_matches_prd(void) {
    struct Row { OrionError code; const char* action; };
    struct Row rows[] = {
        { ORION_OK,                   "continue" },
        { ORION_ERR_COMPILE_FAILED,   "retry" },
        { ORION_ERR_EVAL_FAILED,      "retry" },
        { ORION_ERR_BUDGET_EXHAUSTED, "worker_replace" },
        { ORION_ERR_SURFACE_INVALID,  "retry" },
        { ORION_ERR_PLAN_INVALID,     "request_fail" },
        { ORION_ERR_TIMEOUT,          "worker_replace" },
        { ORION_ERR_WORKER_DIED,      "worker_replace" },
        { ORION_ERR_CANCELLED,        "cancel" },
    };
    for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); i++) {
        if (strcmp(orion_error_recommended_action(rows[i].code), rows[i].action) != 0) {
            return 0;
        }
    }
    return 1;
}

// Codes are stable and contiguous starting at 0 (mirrors the Python IntEnum).
static int test_values_are_stable_and_contiguous(void) {
    return ORION_OK == 0 &&
           ORION_ERR_COMPILE_FAILED == 1 &&
           ORION_ERR_EVAL_FAILED == 2 &&
           ORION_ERR_BUDGET_EXHAUSTED == 3 &&
           ORION_ERR_SURFACE_INVALID == 4 &&
           ORION_ERR_PLAN_INVALID == 5 &&
           ORION_ERR_TIMEOUT == 6 &&
           ORION_ERR_WORKER_DIED == 7 &&
           ORION_ERR_CANCELLED == 8;
}

int main(void) {
    printf("test_orion_error:\n");

    TEST(every_code_has_a_name);
    TEST(every_code_has_an_action);
    TEST(unknown_codes_fail_closed);
    TEST(action_mapping_matches_prd);
    TEST(values_are_stable_and_contiguous);

    printf("\n%d/%d passed\n", g_passed, g_passed + g_failed);
    return g_failed > 0 ? 1 : 0;
}
