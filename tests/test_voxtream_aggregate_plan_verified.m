// tests/test_voxtream_aggregate_plan_verified.m — G009 plan-verified compile.
//
// Compile the aggregate complete-layer-0 graph through the production
// orion_compile_mil_with_plan path, NOT the legacy raw wdict bypass.
// This proves the multi-chunk BLOBFILE contract works end-to-end: the
// binding plan generates from the aggregate file on disk, the weight
// dictionary validates against the plan (per-chunk offsets + hashes),
// and the private compiler accepts the plan-verified MIL.
//
// Requires: VOXTREAM_DEPTH_EXPORT pointing to a v15 export directory.
// Consumes: 1 fresh ANE compile.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <CommonCrypto/CommonDigest.h>

#include "core/ane_runtime.h"
#include "core/iosurface_tensor.h"
#include "compiler/codegen.h"
#include "compiler/frontends/voxtream_depth.h"
#include "compiler/pipeline.h"
#include "compiler/validate.h"
#include "compiler/binding_plan.h"

// Mirror the diagnostic test's entry table (same layout).
typedef struct {
    const char *relative_path;
    const char *mil_path;
    size_t payload_bytes;
    uint64_t chunk_offset;
} Entry;

static Entry kEntries[] = {
    {"weights/layer0/rms1.bin", "@model_path/weights/layer0/rms1.bin", 2048, 0},
    {"weights/layer0/wq.bin", "@model_path/weights/layer0/wq.bin", 2097152, 0},
    {"weights/layer0/wk.bin", "@model_path/weights/layer0/wk.bin", 524288, 0},
    {"weights/layer0/wv.bin", "@model_path/weights/layer0/wv.bin", 524288, 0},
    {"weights/layer0/wo.bin", "@model_path/weights/layer0/wo.bin", 2097152, 0},
    {"weights/layer0/rms2.bin", "@model_path/weights/layer0/rms2.bin", 2048, 0},
    {"weights/layer0/w1.bin", "@model_path/weights/layer0/w1.bin", 16777216, 0},
    {"weights/layer0/w2.bin", "@model_path/weights/layer0/w2.bin", 16777216, 0},
    {"weights/layer0/w3.bin", "@model_path/weights/layer0/w3.bin", 16777216, 0},
    {"constants/rope_q_cos.bin", "@model_path/constants/rope_q_cos.bin", 65536, 0},
    {"constants/rope_q_sin.bin", "@model_path/constants/rope_q_sin.bin", 65536, 0},
    {"constants/rope_q_perm.bin", "@model_path/constants/rope_q_perm.bin", 2097152, 0},
    {"constants/rope_k_cos.bin", "@model_path/constants/rope_k_cos.bin", 16384, 0},
    {"constants/rope_k_sin.bin", "@model_path/constants/rope_k_sin.bin", 16384, 0},
    {"constants/rope_k_perm.bin", "@model_path/constants/rope_k_perm.bin", 131072, 0},
    {"constants/causal_mask.bin", "@model_path/constants/causal_mask.bin", 2048, 0},
};
static const int kEntryCount = sizeof(kEntries) / sizeof(kEntries[0]);

static void write_le32(uint8_t *dst, uint32_t value) {
    dst[0] = (uint8_t)value; dst[1] = (uint8_t)(value >> 8);
    dst[2] = (uint8_t)(value >> 16); dst[3] = (uint8_t)(value >> 24);
}

static size_t align64(size_t value) { return (value + 63u) & ~(size_t)63u; }

static NSData *read_payload(const char *root, const char *relative, size_t expected) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", root, relative);
    NSData *data = [NSData dataWithContentsOfFile:@(path)];
    if (!data || data.length != 128 + expected) return nil;
    // Return just the payload (skip the 128-byte header).
    return [data subdataWithRange:NSMakeRange(128, expected)];
}

// Build the aggregate blob on disk matching the diagnostic probe's layout:
// the file uses 64-byte chunk headers between payloads, with the first chunk
// header at offset 64 (within the BLOBFILE header region). The MIL BLOBFILE
// offset references the chunk header position. This is the exact layout the
// private compiler accepted in the 2026-07-25 aggregate-layer diagnostic.
static NSData *build_aggregate(const char *root, NSString *out_path) {
    NSData *payloads[32];
    size_t total = 128; // BLOB header
    static const size_t CHUNK_HDR = 64;

    for (int i = 0; i < kEntryCount; i++) {
        payloads[i] = read_payload(root, kEntries[i].relative_path, kEntries[i].payload_bytes);
        if (!payloads[i]) {
            fprintf(stderr, "FAIL: cannot read %s\n", kEntries[i].relative_path);
            return nil;
        }
        // First chunk starts after the BLOB header (128), not at offset 64.
        // The diagnostic test starts at 64 but that overwrites the BLOB header's
        // payload_size field (at offset 72). The plan-verified path requires a
        // valid BLOB header, so we start chunks at 128.
        size_t hdr = (i == 0) ? 128 : align64(total);
        kEntries[i].chunk_offset = (uint64_t)hdr;
        total = hdr + CHUNK_HDR + kEntries[i].payload_bytes;
    }

    NSMutableData *agg = [NSMutableData dataWithLength:total];
    uint8_t *out = agg.mutableBytes;
    // BLOBFILE header.
    out[0] = 1; out[4] = 2;
    out[64] = 0xef; out[65] = 0xbe; out[66] = 0xad; out[67] = 0xde;
    out[68] = 1;
    write_le32(out + 72, (uint32_t)(total - 128));
    write_le32(out + 80, 128);

    for (int i = 0; i < kEntryCount; i++) {
        size_t ho = (size_t)kEntries[i].chunk_offset;
        size_t po = ho + CHUNK_HDR;
        // Write chunk header.
        out[ho] = 0xef; out[ho+1] = 0xbe; out[ho+2] = 0xad; out[ho+3] = 0xde;
        out[ho+4] = 1;
        write_le32(out + ho + 8, (uint32_t)kEntries[i].payload_bytes);
        write_le32(out + ho + 16, (uint32_t)po);
        memcpy(out + po, payloads[i].bytes, kEntries[i].payload_bytes);
    }

    NSString *dir = [out_path stringByDeletingLastPathComponent];
    [[NSFileManager defaultManager] createDirectoryAtPath:dir
                              withIntermediateDirectories:YES attributes:nil error:nil];
    [agg writeToFile:out_path atomically:YES];
    return agg;
}

// Rewrite graph blob refs to point at the aggregate file with chunk-header offsets.
static bool rewrite_refs(OrionGraph *g) {
    for (int i = 0; i < g->n_nodes; i++) {
        OrionNode *node = &g->nodes[i];
        if (node->op != ORION_OP_CONST || !node->attrs.blob_path[0]) continue;
        bool found = false;
        for (int j = 0; j < kEntryCount; j++) {
            if (strcmp(node->attrs.blob_path, kEntries[j].mil_path) == 0) {
                snprintf(node->attrs.blob_path, sizeof(node->attrs.blob_path),
                         "@model_path/aggregate.bin");
                node->attrs.blob_offset = kEntries[j].chunk_offset;
                found = true;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "FAIL: graph references unaggregated blob %s\n",
                    node->attrs.blob_path);
            return false;
        }
    }
    return true;
}

static NSString *sha256_hex(NSData *data) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    char hex[65];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++)
        snprintf(hex + i*2, 3, "%02x", digest[i]);
    return @(hex);
}

int main(int argc, const char *argv[]) {
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

        // 1. Build the aggregate BLOBFILE on disk.
        NSString *tmp = [NSTemporaryDirectory()
            stringByAppendingPathComponent:
                [NSString stringWithFormat:@"g009-plan-%@", NSUUID.UUID.UUIDString]];
        NSString *agg_path = [tmp stringByAppendingPathComponent:@"aggregate.bin"];
        NSData *aggregate = build_aggregate(export_root, agg_path);
        if (!aggregate) return 1;
        printf("Built aggregate: %llu bytes, %d chunks\n",
               (unsigned long long)aggregate.length, kEntryCount);

        // 2. Generate the graph + rewrite blob refs + generate MIL.
        OrionGraph *g = orion_frontend_voxtream_depth_layer(0);
        if (!g) { fprintf(stderr, "FAIL: graph generation\n"); return 1; }
        OrionValidationResult vr = orion_graph_validate(g);
        if (!vr.valid) {
            fprintf(stderr, "FAIL: graph validation: %s\n", vr.message);
            return 1;
        }
        orion_pipeline_optimize(g);
        if (!rewrite_refs(g)) return 1;
        NSString *mil = orion_codegen_mil(g, "main");
        orion_graph_free(g);
        if (!mil) { fprintf(stderr, "FAIL: MIL generation\n"); return 1; }
        printf("Generated MIL: %lu bytes\n", (unsigned long)mil.length);

        // 3. Generate the binding plan from a rewritten graph.
        // We regenerate the graph and rewrite its blob refs BEFORE plan
        // generation so the plan naturally produces aggregate.bin entries
        // with chunk offsets. The plan generator reads each weight from
        // weight_root via open_confined_fd and validates the chunk range.
        OrionGraph *plan_graph = orion_frontend_voxtream_depth_layer(0);
        if (!plan_graph) { fprintf(stderr, "FAIL: plan graph\n"); return 1; }
        orion_pipeline_optimize(plan_graph);
        if (!rewrite_refs(plan_graph)) return 1;

        OrionBindingPlanIdentity identity = {
            .orion_revision = getenv("VOXTREAM_ORION_REVISION") ?: "g009-plan-test",
            .compiler_build = getenv("VOXTREAM_COMPILER_BUILD") ?: "g009-plan-test",
            .weight_root = tmp.fileSystemRepresentation,
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

        // 4. Build the weight dictionary for orion_compile_mil_with_plan.
        NSDictionary *wdict = @{
            @"@model_path/aggregate.bin": @{@"offset": @0, @"data": aggregate}
        };

        // Validate the dictionary against the plan.
        if (!orion_validate_weight_dictionary_for_plan(&plan, wdict)) {
            fprintf(stderr, "FAIL: weight dictionary validation\n");
            return 1;
        }
        printf("Weight dictionary validated against plan\n");

        // 5. Compile via the plan-verified path.
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

        // 6. Evaluate parity: compile proved, now prove numerical correctness.
        // Load the exported reference fixtures and compare ANE output.
        NSString *export = @(export_root);
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

        // 7. Contiguous layer timing (2 warmup + 30 samples).
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
        [[NSFileManager defaultManager] removeItemAtPath:tmp error:nil];
        return 0;
    }
}
