// core/orion_compile_ledger.c — durable compile-budget ledger (ORX-408).
//
// Pure C: no ANE calls, no Objective-C. Mirrors the Python
// voxtream.backends.orion.ledger.CompileLedger dataclass field-for-field.

#include "core/orion_error.h"
#include "core/orion_compile_ledger.h"

#include <string.h>

OrionCompileLedger orion_ledger_init(int planned, int threshold) {
    OrionCompileLedger l;
    l.planned = planned;
    l.cached = 0;
    l.successful = 0;
    l.failed = 0;
    l.threshold = threshold;
    l.empirical_limit = ORION_COMPILE_EMPIRICAL_LIMIT;
    return l;
}

void orion_ledger_record_compile(OrionCompileLedger* ledger, int success, int cached) {
    if (!ledger) return;
    if (cached) { ledger->cached++; return; }
    if (success) ledger->successful++;
    else         ledger->failed++;
}

int orion_ledger_remaining(const OrionCompileLedger* ledger) {
    if (!ledger) return 0;
    int left = ledger->threshold - ledger->successful;
    return left > 0 ? left : 0;
}

int orion_ledger_exhausted(const OrionCompileLedger* ledger) {
    if (!ledger) return 1;  // fail closed on a missing ledger
    return ledger->successful >= ledger->threshold;
}

int orion_ledger_to_dict(const OrionCompileLedger* ledger, FILE* out) {
    if (!ledger || !out) return -1;
    return fprintf(out,
        "{\"planned\":%d,\"cached\":%d,\"successful\":%d,\"failed\":%d,"
        "\"threshold\":%d,\"empirical_limit\":%d,\"remaining\":%d,\"exhausted\":%s}",
        ledger->planned, ledger->cached, ledger->successful, ledger->failed,
        ledger->threshold, ledger->empirical_limit,
        orion_ledger_remaining(ledger),
        orion_ledger_exhausted(ledger) ? "true" : "false");
}
