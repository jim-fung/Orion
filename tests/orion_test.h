// tests/orion_test.h — Shared test result transport.
//
// Orion tests previously conflated "capability skip" with "pass": a test that
// could not run (ANE unavailable, model blobs absent) printed "SKIP" and
// returned exit 0, indistinguishable from a test whose assertions executed and
// passed. This header standardizes the three outcomes so the Makefile runner
// and CI can tell them apart.
//
// Exit-code convention (compatible with automake/gnu "77 = skip"):
//   0  PASS — required assertions executed and passed
//   1  FAIL — an assertion failed or an unexpected error occurred
//   77 SKIP — a declared prerequisite (ANE hardware, model asset) is absent,
//            so the required assertions did not execute
//
// Each test is encouraged to print a final machine-readable summary line of the
// form "RESULT: PASS|FAIL|SKIP <passed> <failed> <skipped>" so a runner can
// parse counts even when normal output is suppressed.
#ifndef ORION_TEST_H
#define ORION_TEST_H

#define ORION_TEST_PASS 0
#define ORION_TEST_FAIL 1
#define ORION_TEST_SKIP 77

// Emit a standardized final summary line. Tests with per-case counters should
// call this before returning. Runners grep for "RESULT:" to aggregate counts.
#define ORION_TEST_SUMMARY(passed, failed, skipped) \
    printf("RESULT: %s %d %d %d\n", \
           (failed) > 0 ? "FAIL" : ((skipped) > 0 ? "SKIP" : "PASS"), \
           (passed), (failed), (skipped))

#endif // ORION_TEST_H
