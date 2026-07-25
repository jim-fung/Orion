// test_voxtream_seqpack_timing.m — Measure seq-pack per-eval latency on M5 Pro.
// 
// Compiles the seq-pack depth-layer graph via plan-verified path, then runs
// N timed evals to measure per-eval latency at seq=64 (both CFG branches
// packed into one surface). This is the key number for ORX-207: if seq=64
// per-eval is fast enough, the full depth loop (60 evals) fits the 60ms budget.
//
// Usage: VOXTREAM_DEPTH_EXPORT=<seqpack-export> ./build/tests/test_voxtream_seqpack_timing [samples]
// Default samples=30. Skip when export unset or ANE unavailable.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/graph.h"
#include "compiler/validate.h"
#include "compiler/pipeline.h"
#include "compiler/codegen.h"
#include "compiler/frontends/voxtream_depth.h"
#include "core/ane_runtime.h"
#include "core/iosurface_tensor.h"

static NSString *kBlobPath = @"@model_path/weights/layer0_seqpack.blob";

static bool rewrite_refs(OrionGraph *g, NSDictionary<NSString *, NSNumber *> *offsets) {
    for (int i = 0; i < g->n_nodes; i++) {
        OrionNode *node = &g->nodes[i];
        if (node->op != ORION_OP_CONST || node->attrs.blob_path[0] == '\0') continue;
        NSString *name = [NSString stringWithUTF8String:node->attrs.blob_path];
        NSString *chunk_name = [name.lastPathComponent stringByDeletingPathExtension];
        NSNumber *offset = offsets[chunk_name];
        if (!offset) { fprintf(stderr, "no chunk for %s\n", node->attrs.blob_path); return false; }
        strcpy(node->attrs.blob_path, kBlobPath.UTF8String);
        node->attrs.blob_offset = [offset unsignedLongLongValue];
    }
    return true;
}

static double monotonic_ms(uint64_t start, uint64_t end) {
    static mach_timebase_info_data_t info;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ mach_timebase_info(&info); });
    return (double)(end - start) * (double)info.numer / (double)info.denom / 1e6;
}

static double percentile_sorted(const double *sorted, int n, double frac) {
    if (n == 0) return 0;
    if (n == 1) return sorted[0];
    double pos = frac * (n - 1);
    int lo = (int)pos; int hi = lo + 1; if (hi >= n) hi = n - 1;
    double f = pos - lo;
    return sorted[lo] * (1.0 - f) + sorted[hi] * f;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        const char *export_root = getenv("VOXTREAM_DEPTH_EXPORT");
        if (!export_root || !*export_root) {
            printf("SKIP: set VOXTREAM_DEPTH_EXPORT to a seqpack export\n");
            return 0;
        }
        if (!orion_ane_init()) { printf("SKIP: ANE unavailable\n"); return 0; }

        int samples = (argc > 1) ? atoi(argv[1]) : 30;
        int warmup = 3;
        printf("=== ORX-207 seq-pack timing probe ===\n");
        printf("  samples: %d (warmup %d)\n", samples, warmup);

        NSString *export = [NSString stringWithUTF8String:export_root];

        // Load manifest + blob.
        NSString *mpath = [export stringByAppendingPathComponent:@"weights/layer0_seqpack.manifest.json"];
        NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:
            [NSData dataWithContentsOfFile:mpath] options:0 error:nil];
        NSMutableDictionary<NSString *, NSNumber *> *offsets = [NSMutableDictionary dictionary];
        for (NSDictionary *c in manifest[@"chunks"]) offsets[c[@"name"]] = c[@"offset"];
        NSData *blob = [NSData dataWithContentsOfFile:
            [export stringByAppendingPathComponent:@"weights/layer0_seqpack.blob"]];

        // Build + compile graph (plan-verified path).
        OrionGraph *g = orion_frontend_voxtream_depth_probe_seq_pack_adapter(0, 0, NULL);
        orion_graph_validate(g); orion_pipeline_optimize(g); rewrite_refs(g, offsets);
        NSString *mil = orion_codegen_mil(g, "main"); orion_graph_free(g);

        OrionGraph *pg = orion_frontend_voxtream_depth_probe_seq_pack_adapter(0, 0, NULL);
        orion_pipeline_optimize(pg); rewrite_refs(pg, offsets);
        OrionBindingPlanIdentity identity = {
            .orion_revision = "orx207-timing", .compiler_build = "orx207-timing",
            .weight_root = export.fileSystemRepresentation,
        };
        OrionBindingPlan plan; char err[512] = {0};
        orion_binding_plan_generate_with_identity(pg, "orx207", &identity, &plan, err, sizeof(err));
        orion_graph_free(pg);
        orion_binding_plan_set_mil_identity(&plan, mil.UTF8String);
        orion_binding_plan_validate_finalized(&plan, err, sizeof(err));

        NSDictionary *wdict = @{kBlobPath: @{@"offset": @0, @"data": blob}};
        orion_validate_weight_dictionary_for_plan(&plan, wdict);

        int count_before = orion_compile_count();
        OrionProgram *prog = orion_compile_mil_with_plan(
            mil.UTF8String, &plan, wdict, "orx207-seqpack-timing", err, sizeof(err));
        int compile_delta = orion_compile_count() - count_before;

        if (!prog) {
            printf("FAIL: compile: %s (delta=%d)\n", err, compile_delta);
            return 1;
        }
        printf("  compiled: delta=%d\n", compile_delta);

        // Create IOSurface for [1, 1024, 1, 64] input + output.
        const int CH = 1024, SEQ = 64;
        size_t surf_bytes = (size_t)CH * SEQ * sizeof(_Float16);
        // Minimum 49KB — our surface is 1024*64*2 = 128KB, well above.
        IOSurfaceRef input_surf = orion_tensor_create(CH, SEQ);
        IOSurfaceRef output_surf = orion_tensor_create(CH, SEQ);
        if (!input_surf || !output_surf) {
            printf("FAIL: IOSurface allocation\n");
            return 1;
        }

        // Fill input with finite fp16 data.
        {
            _Float16 *buf = (_Float16 *)IOSurfaceGetBaseAddress(input_surf);
            IOSurfaceLock(input_surf, 0, NULL);
            for (size_t i = 0; i < (size_t)CH * SEQ; i++) buf[i] = (_Float16)(i % 100 * 0.001);
            IOSurfaceUnlock(input_surf, 0, NULL);
        }

        IOSurfaceRef inputs[] = {input_surf};
        IOSurfaceRef outputs[] = {output_surf};

        // Warmup.
        for (int i = 0; i < warmup; i++) {
            orion_eval(prog, inputs, 1, outputs, 1);
        }

        // Timed samples.
        double *times = (double *)malloc(samples * sizeof(double));
        for (int i = 0; i < samples; i++) {
            uint64_t start = mach_absolute_time();
            orion_eval(prog, inputs, 1, outputs, 1);
            uint64_t end = mach_absolute_time();
            times[i] = monotonic_ms(start, end);
        }

        // Sort for percentiles.
        for (int i = 0; i < samples; i++)
            for (int j = i + 1; j < samples; j++)
                if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }

        double mean = 0; for (int i = 0; i < samples; i++) mean += times[i];
        mean /= samples;

        printf("\n  per-eval latency (seq=64, 1 packed surface):\n");
        printf("    mean: %.3f ms\n", mean);
        printf("    p50:  %.3f ms\n", percentile_sorted(times, samples, 0.50));
        printf("    p95:  %.3f ms\n", percentile_sorted(times, samples, 0.95));
        printf("    p99:  %.3f ms\n", percentile_sorted(times, samples, 0.99));
        printf("    max:  %.3f ms\n", times[samples - 1]);

        // Project the full depth loop: 4 layers × 15 steps = 60 evals.
        double p99 = percentile_sorted(times, samples, 0.99);
        double projected_60 = p99 * 60.0;
        printf("\n  projected full-loop (60 evals):\n");
        printf("    p99 × 60 = %.1f ms (budget 60 ms)\n", projected_60);
        printf("    %s\n", projected_60 < 60.0 ? "WITHIN BUDGET" : "OVER BUDGET");

        // Compare to separate-eval (120 evals at ~0.5ms each = 60ms).
        printf("\n  for comparison: separate-eval (120 evals) measured p99=70.8ms on M5 Pro\n");

        orion_release_program(prog);
        orion_tensor_release(input_surf);
        orion_tensor_release(output_surf);
        free(times);
        return 0;
    }
}
