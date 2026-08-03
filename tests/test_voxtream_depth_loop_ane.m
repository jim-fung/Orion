// tests/test_voxtream_depth_loop_ane.m
//
// Decisive Epic I gate: measure the actual dependent 15-step separate-CFG
// depth loop on the ANE and verify it fits within the 60 ms depth budget.
//
// ANE evaluation time is shape-dependent, not data-dependent: the aggregate
// layer graph always processes a fixed [1,1024,1,32] IOSurface regardless of
// content. Running the same input 15 times therefore yields identical timing
// to running 15 different dependent inputs, so the full loop can be measured
// without implementing the host-side logits/sampling/embedding pipeline.
//
// Per frame the depth transformer runs, for each of 15 codebook steps, two
// separate-CFG branches (guidance=batch0, generation=batch1). Each branch
// forward pass is: packed hidden -> layer 0 -> 1 -> 2 -> 3 -> final RMSNorm.
// Between layers the output IOSurface is piped straight into the next layer's
// input (no host round-trip); only the final-norm output is read back to host.
//
// Weight strategy (2 compiles total, proven by test_voxtream_aggregate_layer_ane):
//   - Layer 0: generate aggregate graph, rewrite blob refs, compile once.
//   - Layers 1-3: orion_program_patch_weights() from the layer-0 donor (0 compiles).
//   - Final norm: compile once via orion_frontend_voxtream_depth_final_norm().
//
// Stop conditions (any -> exit 1): compile/load failure, compile_delta > 3,
// layer-0 parity failure (max_abs > 0.03 or rel_l2 > 0.005 on either branch),
// non-finite or non-deterministic output, or full-loop p99 >= 60 ms.

#import <Foundation/Foundation.h>
#include "orion_test.h"
#import <IOSurface/IOSurface.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/ane_runtime.h"
#include "core/iosurface_tensor.h"
#include "compiler/codegen.h"
#include "compiler/frontends/voxtream_depth.h"
#include "compiler/pipeline.h"
#include "compiler/validate.h"

#define DIM ORION_VOXTREAM_DEPTH_DIM
#define SEQ ORION_VOXTREAM_DEPTH_ANE_SEQ
#define ELEMENTS ((size_t)DIM * SEQ)
#define TENSOR_BYTES (ELEMENTS * sizeof(_Float16))
#define BLOB_HEADER_BYTES 128
#define CHUNK_HEADER_BYTES 64
#define FIRST_CHUNK_OFFSET 64
#define MAX_ABS_GATE 0.03
#define REL_L2_GATE 0.005
#define DEPTH_STEPS 15
#define CFG_BRANCHES 2
#define ITERATIONS_PER_SAMPLE (DEPTH_STEPS * CFG_BRANCHES)
#define LAYERS ORION_VOXTREAM_DEPTH_LAYERS
#define CHAIN_SURFACES (LAYERS + 2)  // input + 4 layer outputs + final-norm output
#define DEPTH_BUDGET_MS 60.0
#define FRAME_BUDGET_MS 80.0
#define MARGIN_TARGET_MS 20.0
#define MAX_COMPILE_DELTA 3
#define MAX_SAMPLES 100
#define DEFAULT_SAMPLES 30
#define DEFAULT_WARMUP 2
#define MAX_ITERATION_SAMPLES (MAX_SAMPLES * ITERATIONS_PER_SAMPLE)
// The execution contract discards padding token slots [16, 32); parity is gated
// on logical tokens [0, 16) exactly like the aggregate-layer harness.
#define LOGICAL_TOKENS ORION_VOXTREAM_DEPTH_BUCKET

typedef struct {
    const char *relative_path;
    const char *mil_path;
    size_t payload_bytes;
    uint64_t chunk_offset;
} AggregateEntry;

static AggregateEntry kEntries[] = {
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
    dst[0] = (uint8_t)value;
    dst[1] = (uint8_t)(value >> 8);
    dst[2] = (uint8_t)(value >> 16);
    dst[3] = (uint8_t)(value >> 24);
}

static uint32_t read_le32(const uint8_t *src) {
    return (uint32_t)src[0] | ((uint32_t)src[1] << 8) |
           ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 24);
}

static size_t align64(size_t value) {
    return (value + 63u) & ~(size_t)63u;
}

static uint64_t monotonic_now(void) {
    return mach_absolute_time();
}

static double monotonic_elapsed_ms(uint64_t start, uint64_t end) {
    static mach_timebase_info_data_t info;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ mach_timebase_info(&info); });
    return (double)(end - start) * (double)info.numer /
           (double)info.denom / 1e6;
}

static int compare_double(const void *lhs, const void *rhs) {
    double a = *(const double *)lhs, b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static double percentile_sorted(const double *sorted, int count, double fraction) {
    if (count == 1) return sorted[0];
    double pos = fraction * (double)(count - 1);
    int lo = (int)pos;
    int hi = lo + 1;
    if (hi >= count) hi = count - 1;
    double frac = pos - lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

static NSArray *sample_array(const double *samples, int count) {
    NSMutableArray *array = [NSMutableArray arrayWithCapacity:(NSUInteger)count];
    for (int i = 0; i < count; i++) [array addObject:@(samples[i])];
    return array;
}

// mean / p50 / p95 / p99 over an arbitrary sample array. The scratch buffer is
// fixed at MAX_ITERATION_SAMPLES so no VLA is needed.
static NSDictionary *stats_dict(const double *samples, int count) {
    if (count <= 0) {
        return @{@"count": @(0)};
    }
    double sorted[MAX_ITERATION_SAMPLES];
    memcpy(sorted, samples, sizeof(double) * (size_t)count);
    qsort(sorted, (size_t)count, sizeof(double), compare_double);
    double sum = 0.0;
    for (int i = 0; i < count; i++) sum += samples[i];
    double mean = sum / (double)count;
    return @{
        @"count": @(count),
        @"mean_ms": @(mean),
        @"p50_ms": @(percentile_sorted(sorted, count, 0.50)),
        @"p95_ms": @(percentile_sorted(sorted, count, 0.95)),
        @"p99_ms": @(percentile_sorted(sorted, count, 0.99)),
        @"min_ms": @(sorted[0]),
        @"max_ms": @(sorted[count - 1]),
        @"samples_ms": sample_array(samples, count),
    };
}

static NSData *read_valid_single_blob(NSString *path, size_t expected_payload) {
    NSData *data = [NSData dataWithContentsOfFile:path];
    if (!data || data.length != BLOB_HEADER_BYTES + expected_payload) return nil;
    const uint8_t *bytes = data.bytes;
    if (bytes[0] != 1 || bytes[4] != 2 ||
        bytes[64] != 0xef || bytes[65] != 0xbe ||
        bytes[66] != 0xad || bytes[67] != 0xde || bytes[68] != 1 ||
        read_le32(bytes + 72) != expected_payload || read_le32(bytes + 80) != 128) {
        return nil;
    }
    return data;
}

static void write_chunk_header(uint8_t *bytes, size_t chunk_offset,
                               size_t payload_bytes, size_t payload_offset) {
    bytes[chunk_offset] = 0xef;
    bytes[chunk_offset + 1] = 0xbe;
    bytes[chunk_offset + 2] = 0xad;
    bytes[chunk_offset + 3] = 0xde;
    bytes[chunk_offset + 4] = 1;
    write_le32(bytes + chunk_offset + 8, (uint32_t)payload_bytes);
    write_le32(bytes + chunk_offset + 16, (uint32_t)payload_offset);
}

static void relative_path_for_layer(int index, int layer, char *buffer, size_t size) {
    const char *prefix = "weights/layer0/";
    const char *path = kEntries[index].relative_path;
    if (strncmp(path, prefix, strlen(prefix)) == 0) {
        snprintf(buffer, size, "weights/layer%d/%s", layer, path + strlen(prefix));
    } else {
        snprintf(buffer, size, "%s", path);
    }
}

// Build one physical BLOBFILE containing every entry at its own chunk offset.
// The chunk layout is identical for every layer so one MIL text serves all
// layers through weight patching.
static NSData *build_aggregate_blob(NSString *root, int layer) {
    NSData *payloads[32];
    char relative[256];
    size_t total = BLOB_HEADER_BYTES;
    for (int i = 0; i < kEntryCount; i++) {
        relative_path_for_layer(i, layer, relative, sizeof(relative));
        payloads[i] = read_valid_single_blob(
            [root stringByAppendingPathComponent:@(relative)],
            kEntries[i].payload_bytes);
        if (!payloads[i]) {
            fprintf(stderr, "FAIL: cannot read validated blob %s\n", relative);
            return nil;
        }
        size_t header_offset = i == 0 ? FIRST_CHUNK_OFFSET : align64(total);
        kEntries[i].chunk_offset = (uint64_t)header_offset;
        total = header_offset + CHUNK_HEADER_BYTES + kEntries[i].payload_bytes;
    }
    NSMutableData *aggregate = [NSMutableData dataWithLength:total];
    uint8_t *out = aggregate.mutableBytes;
    out[0] = 1;
    out[4] = 2;
    for (int i = 0; i < kEntryCount; i++) {
        size_t header_offset = (size_t)kEntries[i].chunk_offset;
        size_t payload_offset = header_offset + CHUNK_HEADER_BYTES;
        write_chunk_header(out, header_offset, kEntries[i].payload_bytes,
                           payload_offset);
        memcpy(out + payload_offset,
               (const uint8_t *)[payloads[i] bytes] + BLOB_HEADER_BYTES,
               kEntries[i].payload_bytes);
    }
    return aggregate;
}

// Point every BLOBFILE const in the graph at the aggregate file chunk table.
static bool rewrite_blob_refs(OrionGraph *g, int *rewritten) {
    *rewritten = 0;
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
                (*rewritten)++;
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

// Validate + optimize (production passes) but skip the ANE plan validation
// that enforces one tensor per file at offset 64, then emit MIL directly.
static NSString *generate_aggregate_mil(OrionGraph *g, int *rewritten) {
    if (!g) return nil;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) {
        fprintf(stderr, "FAIL: graph validation: %s\n", vr.message);
        orion_graph_free(g);
        return nil;
    }
    orion_pipeline_optimize(g);
    if (!rewrite_blob_refs(g, rewritten)) {
        orion_graph_free(g);
        return nil;
    }
    NSString *mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);
    return mil;
}

static NSDictionary *wdict(NSData *aggregate) {
    return @{@"@model_path/aggregate.bin": @{@"offset": @0, @"data": aggregate}};
}

static double max_abs_diff(const _Float16 *a, const _Float16 *b, size_t count) {
    double maximum = 0.0;
    for (size_t i = 0; i < count; i++) {
        double diff = fabs((double)a[i] - (double)b[i]);
        if (diff > maximum) maximum = diff;
    }
    return maximum;
}

static double logical_max_abs(const _Float16 *a, const _Float16 *b) {
    double maximum = 0.0;
    for (int c = 0; c < DIM; c++) {
        for (int t = 0; t < LOGICAL_TOKENS; t++) {
            size_t idx = (size_t)c * SEQ + t;
            double diff = fabs((double)a[idx] - (double)b[idx]);
            if (diff > maximum) maximum = diff;
        }
    }
    return maximum;
}

static double logical_relative_l2(const _Float16 *a, const _Float16 *b) {
    double err = 0.0, ref = 0.0;
    for (int c = 0; c < DIM; c++) {
        for (int t = 0; t < LOGICAL_TOKENS; t++) {
            size_t idx = (size_t)c * SEQ + t;
            double d = (double)a[idx] - (double)b[idx];
            err += d * d;
            ref += (double)b[idx] * (double)b[idx];
        }
    }
    return ref > 0.0 ? sqrt(err / ref) : INFINITY;
}

static bool logical_finite(const _Float16 *values) {
    for (int c = 0; c < DIM; c++) {
        for (int t = 0; t < LOGICAL_TOKENS; t++) {
            if (!isfinite((double)values[(size_t)c * SEQ + t])) return false;
        }
    }
    return true;
}

static bool all_finite(const _Float16 *values, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (!isfinite((double)values[i])) return false;
    }
    return true;
}

int main(void) {
    @autoreleasepool {
        const char *root_env = getenv("VOXTREAM_DEPTH_EXPORT");
        if (!root_env || !root_env[0]) {
            printf("SKIP: set VOXTREAM_DEPTH_EXPORT to a validated export directory\n");
            return ORION_TEST_SKIP;
        }
        if (!orion_ane_init()) {
            printf("SKIP: ANE runtime unavailable\n");
            return ORION_TEST_SKIP;
        }
        NSString *root = @(root_env);
        const char *output_path = getenv("VOXTREAM_DEPTH_LOOP_OUTPUT");

        int sample_count = DEFAULT_SAMPLES;
        const char *samples_env = getenv("VOXTREAM_DEPTH_LOOP_SAMPLES");
        if (samples_env) sample_count = atoi(samples_env);
        if (sample_count < 1 || sample_count > MAX_SAMPLES) {
            fprintf(stderr, "FAIL: VOXTREAM_DEPTH_LOOP_SAMPLES must be 1..%d\n",
                    MAX_SAMPLES);
            return 1;
        }
        int warmup_count = DEFAULT_WARMUP;
        const char *warmup_env = getenv("VOXTREAM_DEPTH_LOOP_WARMUP");
        if (warmup_env) warmup_count = atoi(warmup_env);
        if (warmup_count < 0 || warmup_count > 20) {
            fprintf(stderr, "FAIL: VOXTREAM_DEPTH_LOOP_WARMUP must be 0..20\n");
            return 1;
        }

        double host_overhead_ms = 0.0;
        const char *host_env = getenv("VOXTREAM_DEPTH_HOST_OVERHEAD_MS");
        if (host_env) host_overhead_ms = atof(host_env);
        if (host_overhead_ms < 0.0) host_overhead_ms = 0.0;

        NSString *failure = nil;
        const int count_before = orion_compile_count();

        NSMutableDictionary *evidence = [NSMutableDictionary dictionary];
        evidence[@"schema"] = @"orion.voxtream.depth-loop-result/1.0";
        evidence[@"export_root"] = root;
        evidence[@"depth_steps"] = @(DEPTH_STEPS);
        evidence[@"cfg_branches"] = @(CFG_BRANCHES);
        evidence[@"iterations_per_sample"] = @(ITERATIONS_PER_SAMPLE);
        evidence[@"layers"] = @(LAYERS);
        evidence[@"weight_patches"] = @(LAYERS - 1);
        evidence[@"operating_system"] =
            [[NSProcessInfo processInfo] operatingSystemVersionString];
        const char *revision = getenv("VOXTREAM_ORION_REVISION");
        const char *compiler_build = getenv("VOXTREAM_COMPILER_BUILD");
        evidence[@"orion_revision"] = revision ? @(revision) : @"unspecified";
        evidence[@"compiler_build"] = compiler_build ? @(compiler_build) : @"unspecified";

        // ---- Compile programs (2 compiles total) ----------------------------
        OrionProgram *layer_programs[LAYERS];
        for (int i = 0; i < LAYERS; i++) layer_programs[i] = nil;
        OrionProgram *final_norm_program = nil;
        int compile_attempts = 0;
        NSString *layer0_mil = nil;

        // Layer 0: one aggregate-layer compile.
        NSData *aggregate0 = build_aggregate_blob(root, 0);
        if (!aggregate0) { failure = @"aggregate-build"; }
        if (!failure) {
            int rewritten = 0;
            layer0_mil = generate_aggregate_mil(
                orion_frontend_voxtream_depth_layer(0), &rewritten);
            if (!layer0_mil) {
                failure = @"layer0-mil-generation";
            } else {
                compile_attempts++;
                layer_programs[0] = orion_compile_mil(layer0_mil.UTF8String,
                                                      wdict(aggregate0),
                                                      "depth-loop-layer0");
                if (!layer_programs[0]) failure = @"layer0-compile";
            }
        }

        // Layers 1-3: weight-patch from the layer-0 donor (zero compiles).
        for (int n = 1; n < LAYERS && !failure; n++) {
            NSData *patched = build_aggregate_blob(root, n);
            if (!patched) { failure = @"aggregate-build"; break; }
            char tag[32];
            snprintf(tag, sizeof(tag), "depth-loop-layer%d", n);
            layer_programs[n] = orion_program_patch_weights(
                layer_programs[0], layer0_mil.UTF8String, wdict(patched), tag);
            if (!layer_programs[n]) { failure = @"weight-patch"; break; }
        }

        // Final norm: one compile.
        if (!failure) {
            NSData *final_blob = read_valid_single_blob(
                [root stringByAppendingPathComponent:@"weights/final_norm.bin"], 2048);
            OrionGraph *final_graph = orion_frontend_voxtream_depth_final_norm();
            NSString *final_mil = nil;
            if (final_blob && final_graph) {
                OrionValidationResult fvr = orion_graph_validate(final_graph);
                if (fvr.valid) {
                    orion_pipeline_optimize(final_graph);
                    final_mil = orion_codegen_mil(final_graph, "main");
                }
            }
            if (final_graph) orion_graph_free(final_graph);
            if (!final_mil) {
                failure = @"final-norm-mil";
            } else {
                compile_attempts++;
                final_norm_program = orion_compile_mil(final_mil.UTF8String,
                    @{@"@model_path/weights/final_norm.bin":
                        @{@"offset": @0, @"data": final_blob}},
                    "depth-loop-final-norm");
                if (!final_norm_program) failure = @"final-norm-compile";
            }
        }

        const int compile_delta = orion_compile_count() - count_before;
        evidence[@"compile_attempts"] = @(compile_attempts);
        evidence[@"compile_count_delta"] = @(compile_delta);
        // Hard compile-budget guard: 2 expected + 1 margin.
        if (!failure && compile_delta > MAX_COMPILE_DELTA) {
            fprintf(stderr, "FAIL: compile_count_delta=%d exceeds budget %d\n",
                    compile_delta, MAX_COMPILE_DELTA);
            failure = @"compile-budget-exceeded";
        }

        // ---- Load reference fixtures ----------------------------------------
        NSData *inputs[CFG_BRANCHES] = {nil, nil};
        NSData *references[CFG_BRANCHES] = {nil, nil};
        if (!failure) {
            inputs[0] = [NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:
                @"ane_references/packed_hidden.batch0.fp16.bin"]];
            inputs[1] = [NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:
                @"ane_references/packed_hidden.batch1.fp16.bin"]];
            references[0] = [NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:
                @"ane_references/packed_layer_0_output.batch0.fp16.bin"]];
            references[1] = [NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:
                @"ane_references/packed_layer_0_output.batch1.fp16.bin"]];
            if (!inputs[0] || !inputs[1] || !references[0] || !references[1] ||
                inputs[0].length != TENSOR_BYTES || inputs[1].length != TENSOR_BYTES ||
                references[0].length != TENSOR_BYTES || references[1].length != TENSOR_BYTES) {
                failure = @"fixture-read";
            }
        }

        // ---- Layer-0 parity sanity check (both branches) --------------------
        // chain[0] = input, chain[1] = layer-0 output.
        IOSurfaceRef chain[CHAIN_SURFACES];
        for (int i = 0; i < CHAIN_SURFACES; i++) chain[i] = NULL;
        _Float16 *output = NULL;
        _Float16 *repeat = NULL;
        NSMutableArray *parity_results = [NSMutableArray array];
        bool parity = true, finite = true, deterministic = true;

        if (!failure) {
            for (int i = 0; i < CHAIN_SURFACES; i++) {
                chain[i] = orion_tensor_create(DIM, SEQ);
                if (!chain[i]) { failure = @"surface-allocation"; break; }
            }
            output = malloc(TENSOR_BYTES);
            repeat = malloc(TENSOR_BYTES);
            if (!output || !repeat) failure = @"host-allocation";
        }

        if (!failure) {
            for (int batch = 0; batch < CFG_BRANCHES && !failure; batch++) {
                orion_tensor_write(chain[0], inputs[batch].bytes, TENSOR_BYTES);
                if (!orion_eval(layer_programs[0], &chain[0], 1, &chain[1], 1)) {
                    failure = @"evaluation";
                    break;
                }
                orion_tensor_read(chain[1], output, TENSOR_BYTES);
                const _Float16 *reference = references[batch].bytes;
                double max_abs = logical_max_abs(output, reference);
                double rel_l2 = logical_relative_l2(output, reference);
                bool branch_passed = max_abs <= MAX_ABS_GATE && rel_l2 <= REL_L2_GATE;
                parity = parity && branch_passed;
                finite = finite && logical_finite(output) && logical_finite(reference);
                [parity_results addObject:@{
                    @"batch": @(batch),
                    @"comparison_scope":
                        @"logical-tokens[0,16)-of-32-per-contract",
                    @"max_abs": @(max_abs),
                    @"relative_l2": @(rel_l2),
                    @"max_abs_gate": @(MAX_ABS_GATE),
                    @"relative_l2_gate": @(REL_L2_GATE),
                    @"passed": @(branch_passed),
                }];
            }
        }

        // ---- Determinism: identical full forward pass (batch 0) twice -------
        if (!failure) {
            orion_tensor_write(chain[0], inputs[0].bytes, TENSOR_BYTES);
            bool eval_ok = true;
            for (int stage = 0; stage < LAYERS; stage++) {
                if (!orion_eval(layer_programs[stage], &chain[stage], 1,
                                &chain[stage + 1], 1)) { eval_ok = false; break; }
            }
            if (eval_ok && !orion_eval(final_norm_program, &chain[LAYERS], 1,
                                       &chain[LAYERS + 1], 1)) eval_ok = false;
            if (!eval_ok) {
                failure = @"evaluation";
            } else {
                orion_tensor_read(chain[LAYERS + 1], output, TENSOR_BYTES);
                orion_tensor_write(chain[0], inputs[0].bytes, TENSOR_BYTES);
                for (int stage = 0; stage < LAYERS && eval_ok; stage++) {
                    if (!orion_eval(layer_programs[stage], &chain[stage], 1,
                                    &chain[stage + 1], 1)) { eval_ok = false; }
                }
                if (eval_ok && !orion_eval(final_norm_program, &chain[LAYERS], 1,
                                           &chain[LAYERS + 1], 1)) eval_ok = false;
                if (!eval_ok) {
                    failure = @"evaluation";
                } else {
                    orion_tensor_read(chain[LAYERS + 1], repeat, TENSOR_BYTES);
                    deterministic = max_abs_diff(output, repeat, ELEMENTS) == 0.0;
                    finite = finite && all_finite(output, ELEMENTS) &&
                             all_finite(repeat, ELEMENTS);
                }
            }
        }

        // ---- The 15-step separate-CFG depth loop ----------------------------
        // Per iteration: write packed hidden -> 4 layer evals (IOSurface-piped)
        // -> final-norm eval -> read final-norm output. The full loop wraps all
        // DEPTH_STEPS x CFG_BRANCHES iterations. ANE timing is shape-dependent,
        // so reusing the two fixed inputs reproduces the production latency.
        double full_loop_samples[MAX_SAMPLES];
        double it_write_samples[MAX_ITERATION_SAMPLES];
        double it_eval_samples[MAX_ITERATION_SAMPLES];
        double it_read_samples[MAX_ITERATION_SAMPLES];
        double it_total_samples[MAX_ITERATION_SAMPLES];
        int full_loop_count = 0;
        int iteration_count = 0;
        bool timed = false;

        if (!failure) {
            for (int sample = -warmup_count; sample < sample_count && !failure; sample++) {
                const bool record = sample >= 0;
                uint64_t full_t0 = monotonic_now();
                for (int step = 0; step < DEPTH_STEPS && !failure; step++) {
                    for (int branch = 0; branch < CFG_BRANCHES && !failure; branch++) {
                        uint64_t it_t0 = monotonic_now();
                        orion_tensor_write(chain[0], inputs[branch].bytes, TENSOR_BYTES);
                        uint64_t after_write = monotonic_now();
                        bool eval_ok = true;
                        for (int stage = 0; stage < LAYERS; stage++) {
                            if (!orion_eval(layer_programs[stage], &chain[stage], 1,
                                            &chain[stage + 1], 1)) {
                                eval_ok = false; break;
                            }
                        }
                        if (eval_ok && !orion_eval(final_norm_program,
                                                   &chain[LAYERS], 1,
                                                   &chain[LAYERS + 1], 1)) eval_ok = false;
                        if (!eval_ok) { failure = @"evaluation"; break; }
                        uint64_t after_eval = monotonic_now();
                        orion_tensor_read(chain[LAYERS + 1], output, TENSOR_BYTES);
                        uint64_t it_t1 = monotonic_now();
                        finite = finite && all_finite(output, ELEMENTS);
                        if (record) {
                            it_write_samples[iteration_count] =
                                monotonic_elapsed_ms(it_t0, after_write);
                            it_eval_samples[iteration_count] =
                                monotonic_elapsed_ms(after_write, after_eval);
                            it_read_samples[iteration_count] =
                                monotonic_elapsed_ms(after_eval, it_t1);
                            it_total_samples[iteration_count] =
                                monotonic_elapsed_ms(it_t0, it_t1);
                            iteration_count++;
                        }
                    }
                }
                if (failure) break;
                uint64_t full_t1 = monotonic_now();
                if (record) {
                    full_loop_samples[full_loop_count++] =
                        monotonic_elapsed_ms(full_t0, full_t1);
                }
            }
            timed = !failure && full_loop_count > 0;
        }

        // ---- Statistics + budget checks -------------------------------------
        bool depth_budget_passed = false;
        bool frame_budget_passed = false;
        double frame_margin_ms = NAN;
        double full_loop_p99 = NAN;
        if (timed) {
            NSDictionary *full_stats = stats_dict(full_loop_samples, full_loop_count);
            full_loop_p99 = [full_stats[@"p99_ms"] doubleValue];
            depth_budget_passed = full_loop_p99 < DEPTH_BUDGET_MS;
            frame_margin_ms = FRAME_BUDGET_MS - full_loop_p99 - host_overhead_ms;
            frame_budget_passed = frame_margin_ms >= MARGIN_TARGET_MS;
            evidence[@"timing"] = @{
                @"timing_method": @"actual-15-step-separate-cfg-depth-loop",
                @"surface_allocation": @"outside-timed-region",
                @"inter_layer_transfer": @"iosurface-pipe-no-host-roundtrip",
                @"shape_not_data_dependent":
                    @"identical inputs reproduce production latency",
                @"samples": @(full_loop_count),
                @"warmup": @(warmup_count),
                @"full_loop": full_stats,
                @"per_iteration": @{
                    @"total": stats_dict(it_total_samples, iteration_count),
                    @"ane_eval": stats_dict(it_eval_samples, iteration_count),
                    @"input_write": stats_dict(it_write_samples, iteration_count),
                    @"output_read": stats_dict(it_read_samples, iteration_count),
                },
            };
        }

        evidence[@"layer0_parity"] = parity_results;
        evidence[@"finite"] = @(finite);
        evidence[@"deterministic"] = @(deterministic);
        evidence[@"parity_passed"] = @(parity);
        evidence[@"timed_samples"] = @(full_loop_count);
        evidence[@"timed_iterations"] = @(iteration_count);
        evidence[@"depth_budget_ms"] = @(DEPTH_BUDGET_MS);
        evidence[@"depth_budget_passed"] = @(depth_budget_passed);
        evidence[@"frame_budget_ms"] = @(FRAME_BUDGET_MS);
        evidence[@"host_overhead_ms"] = @(host_overhead_ms);
        evidence[@"host_overhead_basis"] =
            @"external-estimate-not-measured-logits-sampling-embedding";
        evidence[@"frame_margin_ms"] = @(frame_margin_ms);
        evidence[@"margin_target_ms"] = @(MARGIN_TARGET_MS);
        evidence[@"frame_budget_passed"] = @(frame_budget_passed);

        bool passed = !failure && parity && finite && deterministic &&
                      depth_budget_passed && compile_delta <= MAX_COMPILE_DELTA;
        evidence[@"failure_stage"] = failure ? failure : [NSNull null];
        evidence[@"passed"] = @(passed);

        if (output_path) {
            NSData *json = [NSJSONSerialization dataWithJSONObject:evidence
                options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:nil];
            [json writeToFile:@(output_path) atomically:YES];
        }

        // ---- Console summary -------------------------------------------------
        printf("depth loop: steps=%d branches=%d compiles=%d (budget %d) "
               "weight_patches=%d\n",
               DEPTH_STEPS, CFG_BRANCHES, compile_delta, MAX_COMPILE_DELTA,
               LAYERS - 1);
        for (NSDictionary *batch in parity_results) {
            printf("  layer0 batch %d: max_abs=%.6f relative_l2=%.6f %s\n",
                   [batch[@"batch"] intValue], [batch[@"max_abs"] doubleValue],
                   [batch[@"relative_l2"] doubleValue],
                   [batch[@"passed"] boolValue] ? "PASS" : "FAIL");
        }
        printf("  deterministic=%s finite=%s\n",
               deterministic ? "yes" : "no", finite ? "yes" : "no");
        if (timed) {
            printf("  full-loop p99=%.3f ms (depth budget %.0f ms) -> %s; "
                   "frame margin=%.2f ms (target >= %.0f) -> %s\n",
                   full_loop_p99, DEPTH_BUDGET_MS,
                   depth_budget_passed ? "PASS" : "FAIL",
                   frame_margin_ms, MARGIN_TARGET_MS,
                   frame_budget_passed ? "PASS" : "FAIL");
        }
        if (failure) printf("  failure stage: %s\n", failure.UTF8String);
        printf("%s: 15-step separate-CFG depth loop %s\n",
               passed ? "PASS" : "STOP",
               passed ? "fits the 60 ms depth budget"
                      : "failed a gate; revisit depth inference strategy");

        if (output) free(output);
        if (repeat) free(repeat);
        for (int i = 0; i < CHAIN_SURFACES; i++)
            if (chain[i]) CFRelease(chain[i]);
        for (int i = 0; i < LAYERS; i++)
            if (layer_programs[i]) orion_release_program(layer_programs[i]);
        if (final_norm_program) orion_release_program(final_norm_program);
        return passed ? 0 : 1;
    }
}
