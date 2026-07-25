// tests/test_orion_ledger.c — ORX-408 compile-budget ledger tests (zero ANE).
//
// Asserts the ledger tracks cached/successful/failed compiles correctly, that
// remaining decreases only on fresh successes, that exhaustion is fail-closed
// at the threshold, and that to_dict emits valid JSON. Pure C.

#include <stdio.h>
#include <string.h>

#include "core/orion_error.h"
#include "core/orion_compile_ledger.h"

static int g_passed = 0, g_failed = 0;

#define TEST(name) do { \
    printf("  %-50s ", #name); \
    if (test_##name()) { printf("PASS\n"); g_passed++; } \
    else { printf("FAIL\n"); g_failed++; } \
} while(0)

// Defaults: zeroed counters, declared threshold/limit, full remaining budget.
static int test_init_defaults(void) {
    OrionCompileLedger l = orion_ledger_init(10, ORION_COMPILE_THRESHOLD_DEFAULT);
    return l.planned == 10 &&
           l.cached == 0 && l.successful == 0 && l.failed == 0 &&
           l.threshold == ORION_COMPILE_THRESHOLD_DEFAULT &&
           l.empirical_limit == ORION_COMPILE_EMPIRICAL_LIMIT &&
           orion_ledger_remaining(&l) == ORION_COMPILE_THRESHOLD_DEFAULT &&
           !orion_ledger_exhausted(&l);
}

// A fresh success consumes one slot from the remaining budget.
static int test_remaining_decreases_on_success(void) {
    OrionCompileLedger l = orion_ledger_init(5, 4);
    int before = orion_ledger_remaining(&l);
    orion_ledger_record_compile(&l, /*success*/ 1, /*cached*/ 0);
    int after = orion_ledger_remaining(&l);
    return after == before - 1 && l.successful == 1 && l.failed == 0 && l.cached == 0;
}

// A failed compile is counted but does not advance `successful` or reduce
// remaining (failed slots are not success slots).
static int test_failed_does_not_advance_successful(void) {
    OrionCompileLedger l = orion_ledger_init(5, 4);
    int before = orion_ledger_remaining(&l);
    orion_ledger_record_compile(&l, /*success*/ 0, /*cached*/ 0);
    return l.failed == 1 && l.successful == 0 &&
           orion_ledger_remaining(&l) == before &&
           !orion_ledger_exhausted(&l);
}

// A cache hit is counted but advances neither successful nor failed.
static int test_cached_does_not_advance_successful(void) {
    OrionCompileLedger l = orion_ledger_init(5, 4);
    int before = orion_ledger_remaining(&l);
    orion_ledger_record_compile(&l, /*success*/ 1, /*cached*/ 1);
    return l.cached == 1 && l.successful == 0 && l.failed == 0 &&
           orion_ledger_remaining(&l) == before;
}

// Exhaustion triggers exactly at threshold and fails closed (remaining clamps 0).
static int test_exhaustion_at_threshold(void) {
    OrionCompileLedger l = orion_ledger_init(5, 3);
    for (int i = 0; i < 3; i++) orion_ledger_record_compile(&l, 1, 0);
    if (!orion_ledger_exhausted(&l)) return 0;
    if (orion_ledger_remaining(&l) != 0) return 0;
    // One more success keeps it exhausted and clamps remaining at 0.
    orion_ledger_record_compile(&l, 1, 0);
    return orion_ledger_exhausted(&l) && orion_ledger_remaining(&l) == 0;
}

// Exhaustion does not trigger one short of the threshold.
static int test_not_exhausted_below_threshold(void) {
    OrionCompileLedger l = orion_ledger_init(5, 3);
    orion_ledger_record_compile(&l, 1, 0);
    orion_ledger_record_compile(&l, 1, 0);
    return !orion_ledger_exhausted(&l) && orion_ledger_remaining(&l) == 1;
}

// Failed compiles do not delay exhaustion: only successes count toward it.
static int test_failures_do_not_count_toward_exhaustion(void) {
    OrionCompileLedger l = orion_ledger_init(5, 2);
    orion_ledger_record_compile(&l, 0, 0);
    orion_ledger_record_compile(&l, 0, 0);
    return !orion_ledger_exhausted(&l) && l.failed == 2 && l.successful == 0;
}

// to_dict emits a non-empty JSON object with every field.
static int test_to_dict_emits_json(void) {
    OrionCompileLedger l = orion_ledger_init(2, 2);
    orion_ledger_record_compile(&l, 1, 0);
    orion_ledger_record_compile(&l, 0, 0);
    orion_ledger_record_compile(&l, 1, 1);

    char buf[512];
    FILE* mem = fmemopen(buf, sizeof(buf), "w");
    if (!mem) return 0;
    int n = orion_ledger_to_dict(&l, mem);
    fclose(mem);
    if (n <= 0) return 0;
    buf[n] = '\0';

    return buf[0] == '{' && buf[n - 1] == '}' &&
           strstr(buf, "\"planned\":2") != NULL &&
           strstr(buf, "\"cached\":1") != NULL &&
           strstr(buf, "\"successful\":1") != NULL &&
           strstr(buf, "\"failed\":1") != NULL &&
           strstr(buf, "\"threshold\":2") != NULL &&
           strstr(buf, "\"remaining\":1") != NULL &&
           strstr(buf, "\"exhausted\":false") != NULL;
}

int main(void) {
    printf("test_orion_ledger:\n");

    TEST(init_defaults);
    TEST(remaining_decreases_on_success);
    TEST(failed_does_not_advance_successful);
    TEST(cached_does_not_advance_successful);
    TEST(exhaustion_at_threshold);
    TEST(not_exhausted_below_threshold);
    TEST(failures_do_not_count_toward_exhaustion);
    TEST(to_dict_emits_json);

    printf("\n%d/%d passed\n", g_passed, g_passed + g_failed);
    return g_failed > 0 ? 1 : 0;
}
