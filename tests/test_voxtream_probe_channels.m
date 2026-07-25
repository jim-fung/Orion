// test_voxtream_probe_channels.m
//
// Bounded channel-count probe for the Orion ANE runtime (ORX-102).
//
// This test determines whether the M5 Pro ANE compiler accepts a graph at a
// given channel count and sequence length. It is the decisive experiment for
// channel-packed CFG: if 2048 channels @ seq=32 compiles, the latency
// redesign (halving ANE evaluations from 2 to 1 per layer per codebook step)
// is viable.
//
// The test uses build_probe_identity, which is weight-free (no BLOBFILE
// needed), honoring VOXTREAM_PROBE_CHANNELS and VOXTREAM_PROBE_SEQ env vars.
// This probes I/O surface / channel-layout support, not conv MAC routing.
//
// Compile budget: exactly 1 compile on success, 0 on failure (per Orion's
// success counter; the private compiler's internal limit may count failed
// attempts separately). The hard guard is compile_delta <= 1.
//
// Usage:
//   VOXTREAM_PROBE_CHANNELS=2048 VOXTREAM_PROBE_SEQ=32 ./build/test_voxtream_probe_channels
//
// Skip behavior: returns 0 (inert) when VOXTREAM_PROBE_CHANNELS is unset or
// the ANE is unavailable, so it does not consume budget under `make test`
// without explicit opt-in.

#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <stdio.h>

#include "compiler/graph.h"
#include "compiler/validate.h"
#include "compiler/pipeline.h"
#include "compiler/codegen.h"
#include "compiler/frontends/voxtream_depth.h"
#include "core/ane_runtime.h"

int main(void) {
    @autoreleasepool {
        // --- gates: opt-in + ANE availability ---
        const char* channels_env = getenv("VOXTREAM_PROBE_CHANNELS");
        if (!channels_env || !*channels_env) {
            printf("SKIP: set VOXTREAM_PROBE_CHANNELS=<N> [VOXTREAM_PROBE_SEQ=<S>] "
                   "to run the channel-count probe\n");
            return 0;
        }
        const int d = atoi(channels_env);
        if (d <= 0) {
            fprintf(stderr, "FAIL: VOXTREAM_PROBE_CHANNELS=%s is not a positive int\n",
                    channels_env);
            return 1;
        }
        const char* seq_env = getenv("VOXTREAM_PROBE_SEQ");
        const int s = (seq_env && *seq_env) ? atoi(seq_env) : 32;

        if (!orion_ane_init()) {
            printf("SKIP: ANE unavailable (private runtime not accessible)\n");
            return 0;
        }

        printf("=== Orion channel-count probe (ORX-102) ===\n");
        printf("  channels (d): %d\n", d);
        printf("  seq (s):      %d\n", s);
        printf("  graph:        identity (weight-free, elementwise add)\n");

        // --- build the probe graph ---
        // build_probe_identity reads VOXTREAM_PROBE_CHANNELS/SEQ from the env
        // and builds: input hidden {1,d,1,s} -> add(hidden,hidden) -> output.
        OrionGraph* graph = orion_frontend_voxtream_depth_probe_identity_adapter(
            0, 0, NULL);
        if (!graph) {
            fprintf(stderr, "FAIL: probe graph build returned NULL\n");
            return 1;
        }

        OrionValidationResult vr = orion_graph_validate(graph);
        if (!vr.valid) {
            fprintf(stderr, "FAIL: graph validation: %s\n",
                    vr.message[0] ? vr.message : "(no message)");
            orion_graph_free(graph);
            return 1;
        }

        orion_pipeline_optimize(graph);

        NSString* mil = orion_codegen_mil(graph, "probe_channels");
        orion_graph_free(graph);
        if (!mil || mil.length == 0) {
            fprintf(stderr, "FAIL: MIL codegen produced empty output\n");
            return 1;
        }

        printf("  MIL size:     %lu bytes\n", (unsigned long)mil.length);

        // --- bounded compile (budget guard: delta <= 1) ---
        const int count_before = orion_compile_count();
        OrionProgram* prog = orion_compile_mil(
            mil.UTF8String, @{}, "probe-channels");
        const int compile_delta = orion_compile_count() - count_before;

        // Hard budget guard: this probe must never consume more than 1 compile.
        if (compile_delta > 1) {
            fprintf(stderr,
                    "FAIL: compile budget guard violated: delta=%d (expected <=1)\n",
                    compile_delta);
            if (prog) orion_release_program(prog);
            return 1;
        }

        // --- report ---
        if (prog) {
            printf("\n");
            printf("PROBE PASS: %d channels @ seq=%d COMPILED (delta=%d)\n",
                   d, s, compile_delta);
            printf("  -> Channel-packed CFG at this dimension is COMPILER-VIABLE.\n");
            orion_release_program(prog);
            return 0;
        } else {
            printf("\n");
            printf("PROBE FAIL: %d channels @ seq=%d REJECTED by ANE compiler "
                   "(delta=%d)\n", d, s, compile_delta);
            printf("  -> Channel-packed CFG at this dimension is NOT viable. "
                   "Fall back to sequence-pack or graph-fusion redesign.\n");
            return 1;
        }
    }
}
