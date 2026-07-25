// test_m5_compile_limit.m — Quick M5 Pro compile limit probe.
// Compiles identical small identity kernels until the compiler fails,
// printing the count. Each compile is identical (ANE caches by MIL text),
// so this tests whether the CACHE prevents counting or not.
// Then compiles UNIQUE kernels (different tags → same MIL → cached?).
//
// Usage: ./build/tests/test_m5_compile_limit [max_iters]
// Default max_iters=300. Skip if ANE unavailable.

#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include "core/ane_runtime.h"
#include "compiler/graph.h"
#include "compiler/validate.h"
#include "compiler/pipeline.h"
#include "compiler/codegen.h"
#include "compiler/frontends/voxtream_depth.h"

int main(int argc, char *argv[]) {
    @autoreleasepool {
        if (!orion_ane_init()) {
            printf("SKIP: ANE unavailable\n");
            return 0;
        }
        int max_iters = (argc > 1) ? atoi(argv[1]) : 300;

        printf("=== M5 Pro Compile Limit Probe ===\n");
        printf("  max_iters: %d\n", max_iters);
        printf("  strategy: compile identical identity graphs with unique tags\n\n");

        int compiled = 0;
        for (int i = 0; i < max_iters; i++) {
            // Build a trivial weight-free graph: input -> add(input,input) -> output.
            OrionGraph *g = orion_frontend_voxtream_depth_probe_identity_adapter(0, 0, NULL);
            if (!g) { printf("FAIL: graph build at iter %d\n", i); break; }
            OrionValidationResult vr = orion_graph_validate(g);
            if (!vr.valid) { orion_graph_free(g); printf("FAIL: validate at iter %d\n", i); break; }
            orion_pipeline_optimize(g);
            char tag[64];
            snprintf(tag, sizeof(tag), "limit-probe-%d", i);
            NSString *mil = orion_codegen_mil(g, "probe_channels");
            orion_graph_free(g);
            if (!mil) { printf("FAIL: codegen at iter %d\n", i); break; }

            // Each compile uses a unique tag → unique MIL text → NOT cached.
            OrionProgram *prog = orion_compile_mil(mil.UTF8String, @{}, tag);
            if (!prog) {
                printf("\n=== COMPILE LIMIT REACHED at iteration %d ===\n", i);
                printf("  orion_compile_count(): %d\n", orion_compile_count());
                printf("  Successfully compiled: %d unique kernels\n", compiled);
                break;
            }
            orion_release_program(prog);
            compiled++;

            if (compiled % 50 == 0) {
                printf("  ... %d compiled (count=%d)\n", compiled, orion_compile_count());
            }
        }

        if (compiled == max_iters) {
            printf("\n=== NO LIMIT HIT in %d iterations ===\n", max_iters);
            printf("  orion_compile_count(): %d\n", orion_compile_count());
        }
        printf("\nDone. M5 Pro compile limit: %s\n",
               compiled < max_iters ? "HIT" : ">= max_iters (raise limit)");
        return 0;
    }
}
