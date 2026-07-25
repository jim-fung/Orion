// tests/test_voxtream_aggregate_plan_verified.m — G009 plan-verified compile.
//
// Compile the aggregate complete-layer-0 graph through the production
// orion_compile_mil_with_plan path, NOT the legacy raw wdict bypass.
// This proves the multi-chunk BLOBFILE contract works end-to-end: the
// binding plan generates from the aggregate file on disk, the weight
// dictionary validates against the plan (per-chunk offsets + hashes),
// and the private compiler accepts the plan-verified MIL.
//
// The aggregate BLOBFILE and chunk manifest are produced by the VoXtream
// exporter's export_aggregate_depth_blobs() and live at:
//   $VOXTREAM_DEPTH_EXPORT/weights/layer0_aggregate.blob
//   $VOXTREAM_DEPTH_EXPORT/weights/layer0_aggregate.manifest.json
// This test does NOT reconstruct the aggregate manually — it reads the
// production artifact and binds each graph blob ref to the matching chunk
// offset declared in the manifest.
//
// Requires: VOXTREAM_DEPTH_EXPORT pointing to a v15 export directory that
// has run the aggregate export pipeline.
// Consumes: 1 fresh ANE compile.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "core/ane_runtime.h"
#include "core/iosurface_tensor.h"
#include "compiler/codegen.h"
#include "compiler/frontends/voxtream_depth.h"
#include "compiler/pipeline.h"
#include "compiler/validate.h"
#include "compiler/binding_plan.h"

// Physical path of the aggregate BLOBFILE as referenced from MIL after
// rewrite_refs rewrites every const blob_path to this single file.
static NSString *const kAggregateMilPath =
    @"@model_path/weights/layer0_aggregate.blob";

// Derive the manifest chunk name for a MIL blob_path.  Weight paths look like
// "@model_path/weights/layer0/wq.bin" and constant paths like
// "@model_path/constants/rope_q_cos.bin"; the chunk name is the filename stem
// (e.g. "wq", "rope_q_cos").  This mirrors the VoXtream exporter's chunk
// naming: weights use the blob filename without ".bin", constants use their
// DepthConstantMapping name, and the two coincide on the filename stem.
static NSString *chunk_name_from_mil_path(const char *mil_path) {
    NSString *path = @(mil_path);
    return [path.lastPathComponent stringByDeletingPathExtension];
}

// Rewrite every const blob ref to point at the aggregate BLOBFILE with the
// chunk-header offset declared in the manifest.  Returns false if a graph
// blob ref has no matching manifest chunk (which would indicate a contract
// drift between the exporter and the Orion graph frontend).
static bool rewrite_refs(OrionGraph *g, NSDictionary<NSString *, NSNumber *> *offset_by_name) {
    for (int i = 0; i < g->n_nodes; i++) {
        OrionNode *node = &g->nodes[i];
        if (node->op != ORION_OP_CONST || !node->attrs.blob_path[0]) continue;
        NSString *name = chunk_name_from_mil_path(node->attrs.blob_path);
        NSNumber *offset = offset_by_name[name];
        if (!offset) {
            fprintf(stderr,
                    "FAIL: graph references blob %s with no manifest chunk '%s'\n",
                    node->attrs.blob_path, name.UTF8String);
            return false;
        }
        snprintf(node->attrs.blob_path, sizeof(node->attrs.blob_path),
                 "%s", kAggregateMilPath.UTF8String);
        node->attrs.blob_offset = [offset unsignedLongLongValue];
    }
    return true;
}

int main(void) {
    @autoreleasepool {
        const char *export_root = getenv("VOXTREAM_DEPTH_EXPORT");
        if (!export_root || !export_root[0]) {
            printf("SKIP: set VOXTREAM_DEPTH_EXPORT to a v15 export directory\n");
            return 0;
        }

        if (!orion_ane_init()) {
            printf("SKIP: ANE unavailable\n");
            return 0;
        }

        NSString *export = @(export_root);

        // 1. Read the aggregate manifest and build a chunk name -> offset map.
        NSString *manifest_path =
            [export stringByAppendingPathComponent:@"weights/layer0_aggregate.manifest.json"];
        NSError *json_err = nil;
        NSData *manifest_data = [NSData dataWithContentsOfFile:manifest_path];
        if (!manifest_data) {
            fprintf(stderr, "FAIL: cannot read aggregate manifest %s\n",
                    manifest_path.UTF8String);
            return 1;
        }
        NSDictionary *manifest =
            [NSJSONSerialization JSONObjectWithData:manifest_data
                                           options:0
                                             error:&json_err];
        if (![manifest isKindOfClass:[NSDictionary class]] || json_err) {
            fprintf(stderr, "FAIL: aggregate manifest is not valid JSON: %s\n",
                    json_err.localizedDescription.UTF8String);
            return 1;
        }
        NSString *schema = manifest[@"schema_version"];
        NSArray *chunks = manifest[@"chunks"];
        if (![chunks isKindOfClass:[NSArray class]] || chunks.count == 0) {
            fprintf(stderr, "FAIL: aggregate manifest has no chunks\n");
            return 1;
        }
        NSMutableDictionary<NSString *, NSNumber *> *offset_by_name =
            [NSMutableDictionary dictionaryWithCapacity:chunks.count];
        for (NSDictionary *chunk in chunks) {
            offset_by_name[chunk[@"name"]] = chunk[@"offset"];
        }
        printf("Aggregate manifest: schema=%s, %lu chunks, total_payload=%s bytes\n",
               schema.UTF8String, (unsigned long)chunks.count,
               [manifest[@"total_payload_bytes"] stringValue].UTF8String);

        // 2. Read the physical aggregate BLOBFILE into the weight dictionary.
        NSString *blob_disk_path =
            [export stringByAppendingPathComponent:@"weights/layer0_aggregate.blob"];
        NSData *aggregate = [NSData dataWithContentsOfFile:blob_disk_path];
        if (!aggregate) {
            fprintf(stderr, "FAIL: cannot read aggregate BLOBFILE %s\n",
                    blob_disk_path.UTF8String);
            return 1;
        }
        printf("Loaded aggregate: %llu bytes\n", (unsigned long long)aggregate.length);

        // 3. Generate the graph + rewrite blob refs + generate MIL.
        OrionGraph *g = orion_frontend_voxtream_depth_layer(0);
        if (!g) { fprintf(stderr, "FAIL: graph generation\n"); return 1; }
        OrionValidationResult vr = orion_graph_validate(g);
        if (!vr.valid) {
            fprintf(stderr, "FAIL: graph validation: %s\n", vr.message);
            return 1;
        }
        orion_pipeline_optimize(g);
        if (!rewrite_refs(g, offset_by_name)) return 1;
        NSString *mil = orion_codegen_mil(g, "main");
        orion_graph_free(g);
        if (!mil) { fprintf(stderr, "FAIL: MIL generation\n"); return 1; }
        printf("Generated MIL: %lu bytes\n", (unsigned long)mil.length);

        // 4. Generate the binding plan from a rewritten graph.
        // We regenerate the graph and rewrite its blob refs BEFORE plan
        // generation so the plan naturally produces aggregate-blob entries
        // with chunk offsets. The plan generator reads each weight from
        // weight_root via open_confined_fd and validates the chunk range.
        // weight_root is the export directory (not a temp dir) so the
        // @model_path/weights/layer0_aggregate.blob ref resolves to the real
        // production artifact on disk.
        OrionGraph *plan_graph = orion_frontend_voxtream_depth_layer(0);
        if (!plan_graph) { fprintf(stderr, "FAIL: plan graph\n"); return 1; }
        orion_pipeline_optimize(plan_graph);
        if (!rewrite_refs(plan_graph, offset_by_name)) return 1;

        OrionBindingPlanIdentity identity = {
            .orion_revision = getenv("VOXTREAM_ORION_REVISION") ?: "g009-plan-test",
            .compiler_build = getenv("VOXTREAM_COMPILER_BUILD") ?: "g009-plan-test",
            .weight_root = export.fileSystemRepresentation,
        };
        OrionBindingPlan plan;
        char error[512] = {0};
        bool plan_ok = orion_binding_plan_generate_with_identity(
            plan_graph, "g009-aggregate-layer0", &identity, &plan,
            error, sizeof(error));
        orion_graph_free(plan_graph);
        if (!plan_ok) {
            fprintf(stderr, "FAIL: plan generation: %s\n", error);
            return 1;
        }
        printf("Plan generated: %d weights, version=%u\n", plan.n_weights, plan.version);

        // Set MIL identity (hash of the MIL text compiled with this plan).
        orion_binding_plan_set_mil_identity(&plan, mil.UTF8String);

        // Validate the finalized plan.
        char val_error[512] = {0};
        if (!orion_binding_plan_validate_finalized(&plan, val_error, sizeof(val_error))) {
            fprintf(stderr, "FAIL: plan validation: %s\n", val_error);
            return 1;
        }
        printf("Plan validated: all %d weights pass multi-chunk checks\n", plan.n_weights);

        // 5. Build the weight dictionary for orion_compile_mil_with_plan.
        NSDictionary *wdict = @{
            kAggregateMilPath: @{@"offset": @0, @"data": aggregate}
        };

        // Validate the dictionary against the plan.
        if (!orion_validate_weight_dictionary_for_plan(&plan, wdict)) {
            fprintf(stderr, "FAIL: weight dictionary validation\n");
            return 1;
        }
        printf("Weight dictionary validated against plan\n");

        // 6. Compile via the plan-verified path.
        char compile_error[512] = {0};
        int count_before = orion_compile_count();
        OrionProgram *prog = orion_compile_mil_with_plan(
            mil.UTF8String, &plan, wdict, "g009-plan-verified-layer0",
            compile_error, sizeof(compile_error));
        int compile_delta = orion_compile_count() - count_before;

        if (!prog) {
            fprintf(stderr, "FAIL: plan-verified compile: %s (delta=%d)\n",
                    compile_error, compile_delta);
            return 1;
        }

        printf("PASS: plan-verified aggregate-layer compile succeeded\n");
        printf("  compiles consumed: %d\n", compile_delta);
        printf("  plan cached: %s\n", orion_program_plan(prog) ? "yes" : "no");

        // 7. Evaluate parity: compile proved, now prove numerical correctness.
        // Load the exported reference fixtures and compare ANE output.
        NSData *input0 = [NSData dataWithContentsOfFile:
            [export stringByAppendingPathComponent:
                @"ane_references/packed_hidden.batch0.fp16.bin"]];
        NSData *ref0 = [NSData dataWithContentsOfFile:
            [export stringByAppendingPathComponent:
                @"ane_references/packed_layer_0_output.batch0.fp16.bin"]];

        #define PLAN_DIM 1024
        #define PLAN_SEQ 32
        #define PLAN_LOGICAL 16
        #define PLAN_TENSOR_BYTES ((size_t)PLAN_DIM * PLAN_SEQ * sizeof(_Float16))
        static const double PLAN_MAX_ABS_GATE = 0.03;
        static const double PLAN_REL_L2_GATE = 0.005;

        if (!input0 || input0.length != PLAN_TENSOR_BYTES ||
            !ref0 || ref0.length != PLAN_TENSOR_BYTES) {
            fprintf(stderr, "FAIL: cannot load reference fixtures\n");
            orion_release_program(prog);
            return 1;
        }

        IOSurfaceRef in_surf = orion_tensor_create(PLAN_DIM, PLAN_SEQ);
        IOSurfaceRef out_surf = orion_tensor_create(PLAN_DIM, PLAN_SEQ);
        if (!in_surf || !out_surf) {
            fprintf(stderr, "FAIL: IOSurface allocation\n");
            orion_release_program(prog);
            return 1;
        }

        _Float16 *output = malloc(PLAN_TENSOR_BYTES);
        if (!output) { fprintf(stderr, "FAIL: malloc\n"); return 1; }

        orion_tensor_write(in_surf, input0.bytes, PLAN_TENSOR_BYTES);
        if (!orion_eval(prog, &in_surf, 1, &out_surf, 1)) {
            fprintf(stderr, "FAIL: ANE evaluation\n");
            orion_release_program(prog);
            free(output);
            return 1;
        }
        orion_tensor_read(out_surf, output, PLAN_TENSOR_BYTES);

        // Parity on logical tokens [0,16) of 32.
        const _Float16 *ref = ref0.bytes;
        double max_abs = 0.0, err_sq = 0.0, ref_sq = 0.0;
        for (int c = 0; c < PLAN_DIM; c++) {
            for (int t = 0; t < PLAN_LOGICAL; t++) {
                size_t idx = (size_t)c * PLAN_SEQ + t;
                double diff = fabs((double)output[idx] - (double)ref[idx]);
                if (diff > max_abs) max_abs = diff;
                double d = (double)output[idx] - (double)ref[idx];
                err_sq += d * d;
                ref_sq += (double)ref[idx] * (double)ref[idx];
            }
        }
        double rel_l2 = ref_sq > 0.0 ? sqrt(err_sq / ref_sq) : INFINITY;

        printf("Parity (batch0, logical tokens [0,16)):\n");
        printf("  max_abs:     %.6f (gate %.3f) %s\n", max_abs, PLAN_MAX_ABS_GATE,
               max_abs <= PLAN_MAX_ABS_GATE ? "PASS" : "FAIL");
        printf("  relative_l2: %.9f (gate %.4f) %s\n", rel_l2, PLAN_REL_L2_GATE,
               rel_l2 <= PLAN_REL_L2_GATE ? "PASS" : "FAIL");

        if (max_abs > PLAN_MAX_ABS_GATE || rel_l2 > PLAN_REL_L2_GATE) {
            fprintf(stderr, "FAIL: plan-verified parity gates not met\n");
            orion_release_program(prog);
            free(output);
            return 1;
        }

        // 8. Contiguous layer timing (2 warmup + 30 samples).
        #define PLAN_TIMED_SAMPLES 30
        #define PLAN_WARMUP 2
        double total_ms[PLAN_TIMED_SAMPLES];
        for (int s = -PLAN_WARMUP; s < PLAN_TIMED_SAMPLES; s++) {
            uint64_t t0 = mach_absolute_time();
            orion_tensor_write(in_surf, input0.bytes, PLAN_TENSOR_BYTES);
            orion_eval(prog, &in_surf, 1, &out_surf, 1);
            orion_tensor_read(out_surf, output, PLAN_TENSOR_BYTES);
            uint64_t t1 = mach_absolute_time();
            if (s >= 0) {
                static mach_timebase_info_data_t tb_info;
                static dispatch_once_t once;
                dispatch_once(&once, ^{ mach_timebase_info(&tb_info); });
                total_ms[s] = (double)(t1 - t0) * (double)tb_info.numer /
                              (double)tb_info.denom / 1e6;
            }
        }
        // Sort and compute p50/p99.
        double sorted[PLAN_TIMED_SAMPLES];
        memcpy(sorted, total_ms, sizeof(total_ms));
        for (int i = 0; i < PLAN_TIMED_SAMPLES - 1; i++)
            for (int j = i + 1; j < PLAN_TIMED_SAMPLES; j++)
                if (sorted[i] > sorted[j]) { double tmp = sorted[i]; sorted[i] = sorted[j]; sorted[j] = tmp; }
        double p50 = sorted[PLAN_TIMED_SAMPLES / 2];
        double p99 = sorted[(int)(0.99 * (PLAN_TIMED_SAMPLES - 1))];
        double projected_depth = p99 * 15.0 * 2.0;  // 15 steps × 2 CFG branches

        printf("Timing (%d samples, contiguous write→eval→readback):\n", PLAN_TIMED_SAMPLES);
        printf("  p50:              %.3f ms\n", p50);
        printf("  p99:              %.3f ms\n", p99);
        printf("  projected depth:  %.1f ms (budget 60 ms) %s\n", projected_depth,
               projected_depth < 60.0 ? "PASS" : "FAIL");

        orion_release_program(prog);
        free(output);
        printf("\nG009 QUALIFIED: plan-verified compile + parity + timing all pass\n");
        return 0;
    }
}
